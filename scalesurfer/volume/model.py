import math
import os
from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PATCH_SIZE
from ..utils import ensure_divisible, volume_to_patches

def _profile_enabled() -> bool:
    return _profile_level() > 0

def _profile_level() -> int:
    value = os.environ.get("SCALESURFER_PROFILE", "")
    key = value.strip().lower()
    if key in {"", "0", "false", "no", "off"}:
        return 0
    try:
        return int(key)
    except ValueError:
        return 1

def _sync_tensor_device(x) -> None:
    if torch.is_tensor(x):
        device = x.device
    else:
        device = torch.device(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

def _group_norm(ch):
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _group_norm(out_ch),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _group_norm(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.block = ConvBlock3D(out_ch, out_ch, dropout=dropout)

    def forward(self, x):
        return self.block(self.down(x))

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.reduce = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.block = ConvBlock3D(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class TransUNet3D(nn.Module):
    def __init__(
        self,
        n_classes,
        in_channels=1,
        base_shape=(208, 240, 192),
        patch_size=PATCH_SIZE,
        channels=(16, 24, 48, 72, 96, 128),
        transformer_depth=2,
        n_heads=4,
        dropout=0.1,
        positional_encoding="learned",
        task_type="classification",
    ):
        super().__init__()
        if len(channels) != 6:
            raise ValueError("channels must have length 6")

        self.n_classes = int(n_classes)
        self.base_shape = tuple(int(v) for v in base_shape)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.task_type = str(task_type).strip().lower()
        if self.task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")
        if self.task_type == "classification" and self.n_classes < 2:
            raise ValueError(f"classification requires n_classes >= 2, got {self.n_classes}")

        c0, c1, c2, c3, c4, c5 = [int(c) for c in channels]

        self.stem = ConvBlock3D(in_channels, c0, dropout=0.0)
        self.down1 = DownBlock3D(c0, c1, dropout=0.0)
        self.down2 = DownBlock3D(c1, c2, dropout=0.0)
        self.down3 = DownBlock3D(c2, c3, dropout=dropout)
        self.down4 = DownBlock3D(c3, c4, dropout=dropout)
        self.down5 = DownBlock3D(c4, c5, dropout=dropout)
        self.bottleneck_conv = ConvBlock3D(c5, c5, dropout=dropout)

        g = list(self.base_shape)
        for _ in range(5):
            g = [(x + 1) // 2 for x in g]
        self.base_bottleneck_grid = tuple(g)
        self.positional_encoding = str(positional_encoding).strip().lower()
        if self.positional_encoding not in {"learned", "sincos"}:
            raise ValueError("positional_encoding must be 'learned' or 'sincos'")

        if self.positional_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, int(math.prod(g)), c5))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_parameter("pos_embed", None)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=c5,
            nhead=n_heads,
            dim_feedforward=4 * c5,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_depth, enable_nested_tensor=False)

        self.up5 = UpBlock3D(c5, c4, c4, dropout=dropout)
        self.up4 = UpBlock3D(c4, c3, c3, dropout=dropout)
        self.up3 = UpBlock3D(c3, c2, c2, dropout=dropout)
        self.up2 = UpBlock3D(c2, c1, c1, dropout=0.0)
        self.up1 = UpBlock3D(c1, c0, c0, dropout=0.0)

        self.token_norm = nn.LayerNorm(c0)
        self.classifier = nn.Linear(c0, self.n_classes if self.task_type == "classification" else 1)

    def _token_positional_embedding(self, btm):
        if self.positional_encoding == "sincos":
            return self._token_positional_embedding_sincos(btm)

        seq_len = btm.shape[2] * btm.shape[3] * btm.shape[4]
        if seq_len == self.pos_embed.shape[1]:
            return self.pos_embed
        pos3 = self.pos_embed.transpose(1, 2).reshape(1, btm.size(1), *self.base_bottleneck_grid)
        pos3 = F.interpolate(pos3, size=btm.shape[2:], mode="trilinear", align_corners=False)
        return pos3.flatten(2).transpose(1, 2)

    def _token_positional_embedding_sincos(self, btm):
        # Deterministic 3D positional encoding with explicit axis coordinates.
        _, c, d, h, w = btm.shape
        device = btm.device

        freq_count = max(1, c // 6)
        freqs = torch.exp(
            torch.linspace(0.0, math.log(64.0), freq_count, device=device, dtype=torch.float32)
        ) * math.pi

        z = torch.linspace(-1.0, 1.0, d, device=device, dtype=torch.float32)
        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, w, device=device, dtype=torch.float32)

        zf = torch.cat([torch.sin(z[:, None] * freqs[None, :]), torch.cos(z[:, None] * freqs[None, :])], dim=1)
        yf = torch.cat([torch.sin(y[:, None] * freqs[None, :]), torch.cos(y[:, None] * freqs[None, :])], dim=1)
        xf = torch.cat([torch.sin(x[:, None] * freqs[None, :]), torch.cos(x[:, None] * freqs[None, :])], dim=1)

        zf = zf[:, None, None, :].expand(d, h, w, 2 * freq_count)
        yf = yf[None, :, None, :].expand(d, h, w, 2 * freq_count)
        xf = xf[None, None, :, :].expand(d, h, w, 2 * freq_count)
        pos = torch.cat([zf, yf, xf], dim=-1)

        if pos.shape[-1] < c:
            pad = torch.zeros(d, h, w, c - pos.shape[-1], device=device, dtype=pos.dtype)
            pos = torch.cat([pos, pad], dim=-1)
        elif pos.shape[-1] > c:
            pos = pos[..., :c]

        return pos.reshape(1, d * h * w, c).to(dtype=btm.dtype)

    def forward_features(self, x):
        # x: [B, 1, D, H, W]
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, D, H, W], got {tuple(x.shape)}")

        profile_layers = _profile_level() >= 2
        if profile_layers:
            _sync_tensor_device(x)
            layer_t0 = perf_counter()
            last_t = layer_t0
            layer_times = []

            def mark(name, tensor):
                nonlocal last_t
                _sync_tensor_device(tensor)
                now = perf_counter()
                layer_times.append((name, now - last_t))
                last_t = now
        else:
            def mark(name, tensor):
                return None

        x0 = self.stem(x)
        mark("stem", x0)
        x1 = self.down1(x0)
        mark("down1", x1)
        x2 = self.down2(x1)
        mark("down2", x2)
        x3 = self.down3(x2)
        mark("down3", x3)
        x4 = self.down4(x3)
        mark("down4", x4)
        x5 = self.down5(x4)
        mark("down5", x5)

        btm = self.bottleneck_conv(x5)
        mark("bottleneck", btm)
        seq = btm.flatten(2).transpose(1, 2)
        pos = self._token_positional_embedding(btm)
        mark("pos", pos)
        seq = self.transformer(seq + pos)
        mark("transformer", seq)
        btm = seq.transpose(1, 2).reshape_as(btm)

        d4 = self.up5(btm, x4)
        mark("up5", d4)
        d3 = self.up4(d4, x3)
        mark("up4", d3)
        d2 = self.up3(d3, x2)
        mark("up3", d2)
        d1 = self.up2(d2, x1)
        mark("up2", d1)
        d0 = self.up1(d1, x0)
        mark("up1", d0)
        if profile_layers:
            total = sum(sec for _, sec in layer_times)
            parts = " ".join(f"{name}={sec:.3f}s" for name, sec in layer_times)
            print(f"[scalesurfer] forward_features: {parts} total={total:.3f}s")
        return d0

    def features_to_patch_features(self, feat):
        # feat: [B, C, D, H, W] -> [B, N, V, C]
        if feat.ndim != 5:
            raise ValueError(f"Expected [B, C, D, H, W], got {tuple(feat.shape)}")
        b, c, d, h, w = feat.shape
        ensure_divisible((d, h, w), self.patch_size)
        pd, ph, pw = self.patch_size
        gd, gh, gw = d // pd, h // ph, w // pw
        x = feat.reshape(b, c, gd, pd, gh, ph, gw, pw)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.reshape(b, gd * gh * gw, pd * ph * pw, c)
        return self.token_norm(x)

    def _apply_token_norm_dense_(self, feat):
        # In-place equivalent of token_norm over the channel dimension for every voxel.
        mean = feat.mean(dim=1, keepdim=True)
        var = feat.var(dim=1, keepdim=True, unbiased=False)
        feat.sub_(mean).mul_(torch.rsqrt(var.add_(self.token_norm.eps)))
        if self.token_norm.elementwise_affine:
            weight = self.token_norm.weight.view(1, -1, 1, 1, 1)
            bias = self.token_norm.bias.view(1, -1, 1, 1, 1)
            feat.mul_(weight).add_(bias)
        return feat

    def _token_norm_dense_flat_chunk(self, feat):
        # Stable channel-wise LayerNorm for [B, C, N] classifier chunks.
        # This mirrors token_norm without forcing the whole dense feature map
        # into fp32 at once.
        work = feat.float() if feat.dtype in {torch.float16, torch.bfloat16} else feat
        mean = work.mean(dim=1, keepdim=True)
        var = work.var(dim=1, keepdim=True, unbiased=False)
        work = (work - mean) * torch.rsqrt(var + self.token_norm.eps)
        if self.token_norm.elementwise_affine:
            weight = self.token_norm.weight.view(1, -1, 1).to(device=work.device, dtype=work.dtype)
            bias = self.token_norm.bias.view(1, -1, 1).to(device=work.device, dtype=work.dtype)
            work = work * weight + bias
        return work.to(dtype=feat.dtype)

    def _classifier_chunk_voxels(self, n_voxels: int, patch_chunk_size) -> int:
        if patch_chunk_size is None:
            return int(n_voxels)
        chunk_voxels = int(patch_chunk_size) * int(math.prod(self.patch_size))
        if chunk_voxels <= 0:
            raise ValueError("patch_chunk_size must be positive or None")
        return chunk_voxels

    def _iter_patch_logits(self, patch_feat, patch_chunk_size=96):
        n = patch_feat.shape[1]
        chunk_size = n if patch_chunk_size is None else int(patch_chunk_size)
        if chunk_size <= 0:
            raise ValueError("patch_chunk_size must be positive or None")
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            chunk_feat = patch_feat[:, s:e]
            chunk_logits = self.classifier(chunk_feat)
            yield s, e, chunk_logits

    def loss_from_features(self, feat, y_volume, patch_chunk_size=96):
        # y_volume: [B, D, H, W]
        y_patch = volume_to_patches(y_volume, self.patch_size)
        patch_feat = self.features_to_patch_features(feat)

        total_num = 0.0
        total_den = 0
        for s, e, logit in self._iter_patch_logits(patch_feat, patch_chunk_size=patch_chunk_size):
            tgt = y_patch[:, s:e]
            if self.task_type == "classification":
                loss = F.cross_entropy(logit.permute(0, 3, 1, 2), tgt.long())
            else:
                pred = logit.squeeze(-1)
                loss = F.mse_loss(pred, tgt.to(dtype=pred.dtype))
            n = int(tgt.numel())
            total_num += loss * n
            total_den += n
        return total_num / max(1, total_den)

    def _predict_volume_unchunked(self, feat):
        feat = self._apply_token_norm_dense_(feat)
        weight = self.classifier.weight[:, :, None, None, None]
        logits = F.conv3d(feat, weight, self.classifier.bias)

        if self.task_type == "classification":
            return logits.argmax(dim=1)

        return logits.squeeze(1)

    def _predict_volume_chunked(self, feat, patch_chunk_size=64):
        b, c, d, h, w = feat.shape
        n_voxels = d * h * w
        chunk_voxels = self._classifier_chunk_voxels(n_voxels, patch_chunk_size)
        feat_flat = feat.flatten(2)
        pred_dtype = torch.int64 if self.task_type == "classification" else feat.dtype
        pred_flat = torch.empty((b, n_voxels), device=feat.device, dtype=pred_dtype)
        weight = self.classifier.weight[:, :, None]
        for s in range(0, n_voxels, chunk_voxels):
            e = min(s + chunk_voxels, n_voxels)
            chunk_feat = self._token_norm_dense_flat_chunk(feat_flat[:, :, s:e])
            logit = F.conv1d(chunk_feat, weight, self.classifier.bias)
            if self.task_type == "classification":
                pred_flat[:, s:e] = logit.argmax(dim=1)
            else:
                pred_flat[:, s:e] = logit.squeeze(1)
        pred = pred_flat.reshape(b, d, h, w)
        return pred

    @torch.inference_mode()
    def predict_volume(self, x, patch_chunk_size=64):
        return self.predict_volume_fast(x, patch_chunk_size=patch_chunk_size).cpu()

    @torch.inference_mode()
    def predict_volume_fast(self, x, patch_chunk_size=64):
        profile = _profile_enabled()
        if profile:
            _sync_tensor_device(x)
            t0 = perf_counter()
        feat = self.forward_features(x)
        if profile:
            _sync_tensor_device(feat)
            features_sec = perf_counter() - t0
            classifier_t0 = perf_counter()
        if patch_chunk_size is None:
            pred = self._predict_volume_unchunked(feat)
        else:
            pred = self._predict_volume_chunked(feat, patch_chunk_size=patch_chunk_size)
        if profile:
            _sync_tensor_device(pred)
            classifier_sec = perf_counter() - classifier_t0
            print(
                "[scalesurfer] volume model: "
                f"features={features_sec:.3f}s "
                f"classifier={classifier_sec:.3f}s "
                f"total={features_sec + classifier_sec:.3f}s"
            )
        return pred
