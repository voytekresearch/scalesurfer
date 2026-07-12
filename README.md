# scalesufer

`scalesurfer` is a repository for fast FreeSurfer inference. The volumetric model uses UNet with a Transformer bottleneck. The stats models predict FreeSurfer measures, including cortical thickness, surface area, curvature, and folding index.

## Installation

```bash
pip install scalesurfer
```

## Usage

See the inference [notebook](https://github.com/voytekresearch/scalesurfer/blob/master/docs/notebooks/03_inference/08_inference.ipynb) for additional settings for faster processing.

```python
from scalesurfer import ScaleSurfer

# Anatomical images
adni_dir = "/home/rph/scalesurfer/data/adni_bids"
subjects = ["sub-002S0559", "sub-002S0619"]

anat_files = [
    f"{adni_dir}/sub-002S0559/ses-20060627/anat/sub-002S0559_ses-20060627_T1w.nii.gz",
    f"{adni_dir}/sub-002S0619/ses-20060601/anat/sub-002S0619_ses-20060601_T1w.nii.gz"
]

# Predict aparc+aseg and stats tables
surfer = ScaleSurfer(anat_files, subjects, "/tmp/scalesurfer_subjects", device="cuda")
surfer.prepare_images()
surfer.predict_volumes()
surfer.plot_volume(subjects[0])
df_stats = surfer.predict_stats()
```

## GPU

All models are implemented with pytorch and inference time depends on GPU. CPU-based inference will be much slower. These models were developed on an NVIDIA card with 32 GB of VRAM and 64 RAM. With this hardware, the settings in the inference [notebook](https://github.com/voytekresearch/scalesurfer/blob/master/docs/notebooks/03_inference/08_inference.ipynb) worked well. Please open an issue if inference fails on your hardware.

