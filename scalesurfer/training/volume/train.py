from dataclasses import dataclass
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .config import DEVICE, PATCH_SIZE
from .utils import ensure_divisible, fit_window_to_shape, gpu_mem_gb, halve_window_shape, make_window_slices


@dataclass
class TrainingRuntime:
    optimizer: object
    scheduler: object
    scheduler_name: str
    scheduler_step_on: str
    scaler: object
    use_amp: bool
    amp_dtype: torch.dtype
    runtime_micro_bs: int
    runtime_patch_chunk: int
    runtime_window: tuple | None
    runtime_stride: tuple | None
    ckpt_path: Path
    grad_clip_norm: float
    device: str
    patch_size: tuple
    enforce_global_attention: bool


def init_training_runtime(model, train_cfg, effective_batch_size=None, device=DEVICE, patch_size=PATCH_SIZE):
    if effective_batch_size is None:
        effective_batch_size = int(train_cfg["effective_batch_size"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    min_lr = float(train_cfg["min_lr"])
    scheduler_name = str(train_cfg.get("lr_scheduler", "plateau")).strip().lower()

    if scheduler_name in {"plateau", "reduce_on_plateau"}:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train_cfg.get("plateau_factor", 0.5)),
            patience=int(train_cfg.get("plateau_patience", 1)),
            min_lr=min_lr,
        )
        scheduler_name = "plateau"
        scheduler_step_on = "epoch_metric"
    elif scheduler_name in {"cosine_warmup", "cosine", "cosine_with_warmup"}:
        warmup_steps = int(train_cfg.get("warmup_steps", 0))
        total_steps_cfg = train_cfg.get("cosine_total_steps", None)
        if total_steps_cfg is None:
            raise ValueError(
                "TRAIN_CFG['cosine_total_steps'] must be set when "
                "TRAIN_CFG['lr_scheduler']='cosine_warmup'"
            )
        total_steps = int(total_steps_cfg)
        if total_steps < 1:
            raise ValueError(f"TRAIN_CFG['cosine_total_steps'] must be >= 1, got {total_steps}")
        if warmup_steps < 0:
            raise ValueError(f"TRAIN_CFG['warmup_steps'] must be >= 0, got {warmup_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(
                "TRAIN_CFG['warmup_steps'] must be smaller than TRAIN_CFG['cosine_total_steps'] "
                f"(got warmup_steps={warmup_steps}, cosine_total_steps={total_steps})"
            )

        base_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]

        def make_lr_lambda(base_lr):
            min_ratio = min(1.0, max(0.0, min_lr / max(base_lr, 1e-12)))

            def lr_lambda(step_idx):
                step = int(step_idx)
                if warmup_steps > 0 and step < warmup_steps:
                    warm = float(step + 1) / float(max(1, warmup_steps))
                    return min_ratio + (1.0 - min_ratio) * warm
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = min(1.0, max(0.0, progress))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1.0 - min_ratio) * cosine

            return lr_lambda

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[make_lr_lambda(lr0) for lr0 in base_lrs],
        )
        scheduler_name = "cosine_warmup"
        scheduler_step_on = "step"
    else:
        raise ValueError(
            "TRAIN_CFG['lr_scheduler'] must be one of "
            "{'plateau', 'cosine_warmup'}"
        )

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    runtime_micro_bs = min(int(train_cfg["initial_micro_batch_size"]), int(effective_batch_size))
    runtime_patch_chunk = int(train_cfg["patch_chunk_size"])
    runtime_window = tuple(train_cfg["spatial_window"]) if train_cfg["spatial_window"] is not None else None
    runtime_stride = tuple(train_cfg["spatial_stride"]) if train_cfg["spatial_stride"] is not None else None
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    enforce_global_attention = bool(train_cfg.get("enforce_global_attention", False))
    if enforce_global_attention:
        # Global attention requires a single full-volume window.
        runtime_window = None
        runtime_stride = None

    ckpt_path = Path(train_cfg["checkpoint_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    return TrainingRuntime(
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_name=scheduler_name,
        scheduler_step_on=scheduler_step_on,
        scaler=scaler,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        runtime_micro_bs=runtime_micro_bs,
        runtime_patch_chunk=runtime_patch_chunk,
        runtime_window=runtime_window,
        runtime_stride=runtime_stride,
        ckpt_path=ckpt_path,
        grad_clip_norm=grad_clip_norm,
        device=str(device),
        patch_size=tuple(int(v) for v in patch_size),
        enforce_global_attention=enforce_global_attention,
    )


def train_step_runtime(model, runtime, x_cpu, y_cpu):
    (
        batch_num,
        batch_den,
        hit_oom,
        runtime.runtime_micro_bs,
        runtime.runtime_patch_chunk,
        runtime.runtime_window,
    ) = train_step_adaptive(
        model=model,
        optimizer=runtime.optimizer,
        scaler=runtime.scaler,
        x_cpu=x_cpu,
        y_cpu=y_cpu,
        runtime_micro_bs=runtime.runtime_micro_bs,
        runtime_patch_chunk=runtime.runtime_patch_chunk,
        runtime_window=runtime.runtime_window,
        runtime_stride=runtime.runtime_stride,
        amp_dtype=runtime.amp_dtype,
        use_amp=runtime.use_amp,
        grad_clip_norm=runtime.grad_clip_norm,
        device=runtime.device,
        patch_size=runtime.patch_size,
        enforce_global_attention=runtime.enforce_global_attention,
    )
    if runtime.scheduler_step_on == "step":
        runtime.scheduler.step()
    return batch_num, batch_den, hit_oom


def step_scheduler_epoch(runtime, val_metric):
    # Plateau-style schedulers are stepped with validation metric at epoch end.
    if runtime.scheduler_step_on == "epoch_metric":
        runtime.scheduler.step(float(val_metric))


def evaluate_runtime(model, loader, runtime, max_batches=None, desc="val"):
    return evaluate(
        model=model,
        loader=loader,
        runtime_micro_bs=runtime.runtime_micro_bs,
        runtime_patch_chunk=runtime.runtime_patch_chunk,
        runtime_window=runtime.runtime_window,
        runtime_stride=runtime.runtime_stride,
        amp_dtype=runtime.amp_dtype,
        use_amp=runtime.use_amp,
        enforce_global_attention=runtime.enforce_global_attention,
        max_batches=max_batches,
        desc=desc,
        device=runtime.device,
        patch_size=runtime.patch_size,
    )

def batch_loss_global(model, xb, yb, patch_chunk_size, patch_size=PATCH_SIZE):
    # xb: [B, D, H, W], yb: [B, D, H, W]
    x5 = xb.unsqueeze(1).float()  # [B,1,D,H,W]
    spatial_shape = tuple(int(v) for v in x5.shape[2:])
    ensure_divisible(spatial_shape, patch_size)
    feat = model.forward_features(x5)
    return model.loss_from_features(feat, yb, patch_chunk_size=patch_chunk_size)


def batch_loss_on_windows(
    model,
    xb,
    yb,
    patch_chunk_size,
    requested_window,
    requested_stride,
    patch_size=PATCH_SIZE,
):
    # xb: [B, D, H, W], yb: [B, D, H, W]
    x5 = xb.unsqueeze(1).float()  # [B,1,D,H,W]
    spatial_shape = tuple(int(v) for v in x5.shape[2:])
    ensure_divisible(spatial_shape, patch_size)

    win = fit_window_to_shape(spatial_shape, requested_window, patch_size)
    stride_in = requested_stride if requested_stride is not None else win
    stride = fit_window_to_shape(spatial_shape, stride_in, patch_size)
    stride = tuple(min(s, w) for s, w in zip(stride, win))

    slices = make_window_slices(spatial_shape, win, stride)
    total = 0.0
    for zs, ys, xs in slices:
        xw = x5[:, :, zs, ys, xs]
        yw = yb[:, zs, ys, xs]
        feat = model.forward_features(xw)
        loss_w = model.loss_from_features(feat, yw, patch_chunk_size=patch_chunk_size)
        total = total + loss_w
    return total / max(1, len(slices))

def train_step_adaptive(
    model,
    optimizer,
    scaler,
    x_cpu,
    y_cpu,
    runtime_micro_bs,
    runtime_patch_chunk,
    runtime_window,
    runtime_stride,
    amp_dtype,
    use_amp,
    grad_clip_norm,
    device=DEVICE,
    patch_size=PATCH_SIZE,
    enforce_global_attention=False,
):
    # Returns: (loss_num, loss_den, hit_oom, runtime_micro_bs, runtime_patch_chunk, runtime_window)
    hit_oom = False
    if bool(enforce_global_attention):
        runtime_window = None
        runtime_stride = None

    total_voxels = int(y_cpu.numel())
    while True:
        optimizer.zero_grad(set_to_none=True)
        batch_num = 0.0

        try:
            for s in range(0, x_cpu.size(0), runtime_micro_bs):
                xb = x_cpu[s : s + runtime_micro_bs].to(device, non_blocking=True)
                yb = y_cpu[s : s + runtime_micro_bs].to(device, non_blocking=True)

                weight = float(yb.numel()) / float(total_voxels)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    if bool(enforce_global_attention):
                        loss_micro = batch_loss_global(
                            model=model,
                            xb=xb,
                            yb=yb,
                            patch_chunk_size=runtime_patch_chunk,
                            patch_size=patch_size,
                        )
                    else:
                        loss_micro = batch_loss_on_windows(
                            model=model,
                            xb=xb,
                            yb=yb,
                            patch_chunk_size=runtime_patch_chunk,
                            requested_window=runtime_window,
                            requested_stride=runtime_stride,
                            patch_size=patch_size,
                        )
                    loss_scaled = loss_micro * weight

                if scaler.is_enabled():
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                batch_num += float(loss_micro.detach().item()) * int(yb.numel())

            clip_norm = float(grad_clip_norm)
            if clip_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            return batch_num, total_voxels, hit_oom, runtime_micro_bs, runtime_patch_chunk, runtime_window

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

            hit_oom = True
            if device == "cuda":
                torch.cuda.empty_cache()

            if runtime_micro_bs > 1:
                runtime_micro_bs = max(1, runtime_micro_bs // 2)
                print(f"[OOM] micro_batch_size -> {runtime_micro_bs}")
            elif runtime_patch_chunk > 16:
                runtime_patch_chunk = max(16, runtime_patch_chunk // 2)
                print(f"[OOM] patch_chunk_size -> {runtime_patch_chunk}")
            else:
                if bool(enforce_global_attention):
                    raise RuntimeError(
                        "OOM at minimum micro-batch/patch-chunk while global attention is enforced. "
                        "Reduce model size or disable TRAIN_CFG['enforce_global_attention']."
                    ) from e
                if runtime_window is None:
                    runtime_window = fit_window_to_shape(tuple(int(v) for v in x_cpu.shape[1:]), x_cpu.shape[1:], patch_size)
                new_window = halve_window_shape(runtime_window, patch_size)
                if new_window == runtime_window:
                    raise RuntimeError("OOM at minimum micro-batch/patch-chunk/window settings") from e
                runtime_window = new_window
                print(f"[OOM] spatial_window -> {runtime_window}")

@torch.no_grad()
def evaluate(
    model,
    loader,
    runtime_micro_bs,
    runtime_patch_chunk,
    runtime_window,
    runtime_stride,
    amp_dtype,
    use_amp,
    enforce_global_attention=False,
    max_batches=None,
    desc="val",
    device=DEVICE,
    patch_size=PATCH_SIZE,
):
    model.eval()
    if bool(enforce_global_attention):
        runtime_window = None
        runtime_stride = None
    total_num = 0.0
    total_den = 0

    limit = len(loader) if max_batches is None else min(len(loader), int(max_batches))
    pbar = tqdm(loader, total=limit, desc=desc, leave=False)

    for b_idx, (x_cpu, y_cpu) in enumerate(pbar):
        if max_batches is not None and b_idx >= int(max_batches):
            break

        batch_num = 0.0
        for s in range(0, x_cpu.size(0), runtime_micro_bs):
            xb = x_cpu[s : s + runtime_micro_bs].to(device, non_blocking=True)
            yb = y_cpu[s : s + runtime_micro_bs].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                if bool(enforce_global_attention):
                    loss_micro = batch_loss_global(
                        model=model,
                        xb=xb,
                        yb=yb,
                        patch_chunk_size=runtime_patch_chunk,
                        patch_size=patch_size,
                    )
                else:
                    loss_micro = batch_loss_on_windows(
                        model=model,
                        xb=xb,
                        yb=yb,
                        patch_chunk_size=runtime_patch_chunk,
                        requested_window=runtime_window,
                        requested_stride=runtime_stride,
                        patch_size=patch_size,
                    )
            batch_num += float(loss_micro.item()) * int(yb.numel())

        total_num += batch_num
        total_den += int(y_cpu.numel())

        alloc_gb, peak_gb = gpu_mem_gb()
        pbar.set_postfix(
            loss=f"{(total_num / max(1, total_den)):.4f}",
            mb=runtime_micro_bs,
            pc=runtime_patch_chunk,
            vram=f"{alloc_gb:.1f}/{peak_gb:.1f}G",
        )

    pbar.close()
    model.train()
    return total_num / max(1, total_den)
