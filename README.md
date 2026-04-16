## scalesufer

`scalesurfer` is a repository for fast FreeSurfer inference. The volumetric model uses UNet (local scale), with a Transformer bottlenck (global scale). The volumetric predictions are passed to CortexODE to predict surfaces. The volumetric model takes a few hundred millseconds per image. The surface model takes a few seconds per image. Downstream, FreeSurfer can be used to convert predictions to statistics, e.g., volume, surface area, curvature, thickness features.

## todos

- huggingface: move pre-trained models
- use FreeSurfer to compute stats, etc., from aparc+aseg, ?h.pial, ?h.white predictions:
```
# per-hemisphere surface continuation from existing ?h.white / ?h.pial
recon-all -s <subjid> -hemi lh -smooth2 -inflate2 -curvHK -curvstats -sphere -surfreg -jacobian_white -avgcurv -cortparc -parcstats
recon-all -s <subjid> -hemi rh -smooth2 -inflate2 -curvHK -curvstats -sphere -surfreg -jacobian_white -avgcurv -cortparc -parcstats

# whole-brain ribbon / volumetric outputs and stats
recon-all -s <subjid> -cortribbon -hyporelabel -aparc2aseg -segstats -wmparc

# optional extra cortical atlases/stats
recon-all -s <subjid> -hemi lh -cortparc2 -parcstats2 -cortparc3 -parcstats3
recon-all -s <subjid> -hemi rh -cortparc2 -parcstats2 -cortparc3 -parcstats3

# optional percent-contrast surfaces
recon-all -s <subjid> -hemi lh -pctsurfcon
recon-all -s <subjid> -hemi rh -pctsurfcon
```
- analysis of how close stats above are to FreeSurfer derived
- analysis of timing versus FreeSurfer
- enumerate and compare outputs from above to FreeSurfer recon-all directories, what are we missing?
- tests
- check that training scripts are reproducible, these were transplanted from another repo
- check that cache directories are minimal, only cache what is needed
- check what cortexode was trained on - sometimes it fails. this is likely from passing raw images into the model - i think it expects orig.mgz, a preprocessed version of rawavg.mgz
