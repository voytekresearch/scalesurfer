# scalesufer

`scalesurfer` is a repository for fast FreeSurfer inference. The volumetric model uses UNet (local scale), with a Transformer bottlenck (global scale). The stats models predict FreeSurfer stats.

## Usage

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

See the inference [notebook](https://github.com/voytekresearch/scalesurfer/blob/master/docs/notebooks/03_inference/08_inference.ipynb) for additional settings for faster processing.
