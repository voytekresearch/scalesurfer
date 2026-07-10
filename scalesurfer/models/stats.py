from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import default_aparc_aseg_label_values
from .volume import TransUNet3D
from ..stats.utils import strip_compile_prefix


def load_transunet_from_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> tuple[TransUNet3D, dict]:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = dict(ckpt.get("model_cfg", {}))
    model = TransUNet3D(
        n_classes=int(ckpt.get("n_classes", 118)),
        in_channels=int(model_cfg.get("in_channels", 1)),
        base_shape=tuple(int(v) for v in ckpt.get("base_volume_shape", (256, 256, 256))),
        patch_size=tuple(int(v) for v in ckpt.get("patch_size", (16, 16, 16))),
        channels=tuple(int(v) for v in model_cfg.get("channels", (12, 20, 32, 48, 64, 96))),
        transformer_depth=int(model_cfg.get("transformer_depth", 2)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        positional_encoding=str(model_cfg.get("positional_encoding", "sincos")),
        task_type=str(model_cfg.get("task_type", "classification")),
    )
    state = strip_compile_prefix(ckpt.get("model_state", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_transunet] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load_transunet] unexpected keys: {len(unexpected)}")
    return model.to(device), ckpt


class TransUNetEncoderAdapter(nn.Module):
    def __init__(self, pretrained: TransUNet3D):
        super().__init__()
        self.backbone = pretrained

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"Expected T1 tensor [B,1,D,H,W], got {tuple(x.shape)}")
        b = self.backbone
        x0 = b.stem(x)
        x1 = b.down1(x0)
        x2 = b.down2(x1)
        x3 = b.down3(x2)
        x4 = b.down4(x3)
        x5 = b.down5(x4)
        z = b.bottleneck_conv(x5)
        seq = z.flatten(2).transpose(1, 2)
        seq = b.transformer(seq + b._token_positional_embedding(z))
        z = seq.transpose(1, 2).reshape_as(z)
        return {
            "enc1": x1,
            "enc2": x2,
            "enc3": x3,
            "enc4": x4,
            "z": z,
            "global": z.mean(dim=(2, 3, 4)),
        }

def infer_encoder_channels(encoder: TransUNetEncoderAdapter, device: str | torch.device = "cpu") -> dict[str, int]:
    was_training = encoder.training
    encoder.eval()
    base_shape = tuple(int(v) for v in encoder.backbone.base_shape)
    probe_shape = tuple(max(32, min(64, v)) for v in base_shape)
    with torch.no_grad():
        x = torch.zeros((1, 1, *probe_shape), dtype=torch.float32, device=device)
        features = encoder(x)
    encoder.train(was_training)
    return {name: int(value.shape[1]) for name, value in features.items() if value.ndim == 5}


def _seg_to_label_indices(seg: torch.Tensor, label_lookup: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = seg.flatten(1).to(torch.int64)
    idx = torch.full_like(flat, -1)
    in_range = (flat >= 0) & (flat < int(label_lookup.numel()))
    if in_range.any():
        idx[in_range] = label_lookup[flat[in_range]]
    valid = idx >= 0
    return idx, valid


def _label_lookup_tensor(label_ids: Iterable[int]) -> torch.Tensor:
    labels = [int(v) for v in label_ids]
    max_label = max(labels) if labels else 0
    lookup = torch.full((max_label + 1,), -1, dtype=torch.long)
    for i, label in enumerate(labels):
        if label >= 0:
            lookup[label] = int(i)
    return lookup


REGION_POOL_STAT_NAMES = ("mean", "sum", "std")
LABEL_SIZE_FEATURE_NAMES: tuple[str, ...] = ()
FEATURE_SCHEMA = "region_mean_sum_std_v1"


def masked_pool_stats_by_label(
    features: torch.Tensor,
    seg: torch.Tensor,
    label_lookup: torch.Tensor,
    n_labels: int,
) -> torch.Tensor:
    if seg.ndim != 4:
        raise ValueError(f"Expected segmentation [B,D,H,W], got {tuple(seg.shape)}")
    seg_rs = F.interpolate(seg[:, None].float(), size=features.shape[2:], mode="nearest").squeeze(1).to(torch.int64)
    label_idx, valid = _seg_to_label_indices(seg_rs, label_lookup)
    features_for_pool = features.float()
    bsz, channels = int(features_for_pool.shape[0]), int(features_for_pool.shape[1])
    feat_flat = features_for_pool.flatten(2)
    sum_ = torch.zeros((bsz, int(n_labels), channels), device=features.device, dtype=torch.float32)
    sum_sq = torch.zeros_like(sum_)
    denom = torch.zeros((bsz, int(n_labels)), device=features.device, dtype=torch.float32)
    for b in range(bsz):
        valid_b = valid[b]
        if not valid_b.any():
            continue
        labels_b = label_idx[b, valid_b]
        values_b = feat_flat[b, :, valid_b].transpose(0, 1).contiguous()
        sum_[b].index_add_(0, labels_b, values_b)
        sum_sq[b].index_add_(0, labels_b, values_b.square())
        denom[b].index_add_(0, labels_b, torch.ones_like(labels_b, dtype=denom.dtype))
    denom_safe = denom.clamp_min(1.0)[:, :, None]
    mean = sum_ / denom_safe
    var = (sum_sq / denom_safe) - mean.square()
    std = torch.sqrt(var.clamp_min(0.0) + 1e-6)
    present = (denom > 0)[:, :, None]
    std = torch.where(present, std, torch.zeros_like(std))
    pooled = torch.cat([mean, sum_, std], dim=2)
    return pooled.flatten(1)


class StatsPredictionModel(nn.Module):
    def __init__(
        self,
        encoder: TransUNetEncoderAdapter,
        *,
        label_ids: Iterable[int],
        out_dims: dict[str, int],
        pool_features: tuple[str, ...] = ("enc2", "enc3", "enc4", "z"),
        hidden: int = 256,
        dropout: float = 0.1,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.encoder = encoder
        self.label_ids = [int(v) for v in label_ids]
        self.register_buffer("label_ids_tensor", torch.tensor(self.label_ids, dtype=torch.long), persistent=False)
        self.register_buffer("label_lookup", _label_lookup_tensor(self.label_ids), persistent=False)
        self.pool_features = tuple(pool_features)
        self.pool_stat_names = tuple(REGION_POOL_STAT_NAMES)
        self.label_size_feature_names = tuple(LABEL_SIZE_FEATURE_NAMES)
        self.feature_schema = FEATURE_SCHEMA
        self._encoder_frozen = False
        channels = infer_encoder_channels(encoder, device=device)
        missing = [name for name in self.pool_features if name not in channels]
        if missing:
            raise KeyError(f"Unknown encoder feature names: {missing}")
        pooled_dim = len(self.label_ids) * sum(channels[name] for name in self.pool_features) * len(self.pool_stat_names)
        global_dim = channels["z"]
        in_dim = pooled_dim + global_dim
        self.input_dim = int(in_dim)
        self.heads = nn.ModuleDict()
        for group, out_dim in out_dims.items():
            self.heads[group] = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, int(hidden)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden), int(out_dim)),
            )

    def freeze_encoder(self, frozen: bool = True) -> None:
        self._encoder_frozen = bool(frozen)
        for param in self.encoder.parameters():
            param.requires_grad = not bool(frozen)
        if frozen:
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_encoder_frozen", False):
            self.encoder.eval()
        return self

    def extract_features(self, t1: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        features = self.encoder(t1)
        n_labels = int(self.label_ids_tensor.numel())
        pooled = [masked_pool_stats_by_label(features[name], seg, self.label_lookup, n_labels) for name in self.pool_features]
        pooled.append(features["global"].to(dtype=pooled[0].dtype))
        return torch.cat(pooled, dim=1)

    def forward_from_features(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {group: head(h) for group, head in self.heads.items()}

    def forward(self, t1: torch.Tensor, seg: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_from_features(self.extract_features(t1, seg))


def build_stats_model(
    *,
    segmentation_checkpoint: str | Path,
    out_dims: dict[str, int],
    label_ids: Iterable[int] | None = None,
    pool_features: tuple[str, ...] = ("enc2", "enc3", "enc4", "z"),
    hidden: int = 256,
    dropout: float = 0.1,
    device: str | torch.device = "cpu",
) -> tuple[StatsPredictionModel, dict]:
    transunet, ckpt = load_transunet_from_checkpoint(segmentation_checkpoint, device=device)
    encoder = TransUNetEncoderAdapter(transunet)
    model = StatsPredictionModel(
        encoder,
        label_ids=list(default_aparc_aseg_label_values() if label_ids is None else label_ids),
        out_dims=out_dims,
        pool_features=pool_features,
        hidden=hidden,
        dropout=dropout,
        device=device,
    )
    return model.to(device), ckpt
