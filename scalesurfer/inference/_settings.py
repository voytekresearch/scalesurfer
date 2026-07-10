"""Internal settings for the high-level inference API."""

import torch


VOLUME_MODEL_FILENAME = "transunet3d.safetensors"
DEFAULT_VOLUME_HF_NAMESPACE = "rphammonds"
VOLUME_MODEL_CONFIG = {
    "n_classes": 118,
    "in_channels": 1,
    "base_shape": (256, 256, 256),
    "patch_size": (16, 16, 16),
    "channels": (12, 20, 32, 48, 64, 96),
    "transformer_depth": 2,
    "n_heads": 4,
    "dropout": 0.0,
    "positional_encoding": "sincos",
    "task_type": "classification",
}
VOLUME_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "f32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "f16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}
VOLUME_MODEL_SPECS = {
    5: {"repo_name": "scalesurfer-v5"},
    6: {"repo_name": "scalesurfer-v6"},
    7: {"repo_name": "scalesurfer-v7"},
    8: {"repo_name": "scalesurfer-v8"},
}
VOLUME_PROVENANCE_FILENAME = "scalesurfer_aparc_aseg.json"
VOLUME_LOG_FILENAME = "scalesurfer_aparc_aseg.log"

