# Goldilocks DPM Demos

Demonstrations of the [Goldilocks-DPM](../) framework for data-driven **D**isease **P**rogression **M**odel configuration.

## Contents

1. [goldilocks-pysustain.py](./goldilocks-pysustain.py): **Su**btype and **Sta**ge **In**ference ([Young et al., Nature Communications 2018](https://doi.org/10.1038/s41467-018-05892-0))
1. Future feature (**TODO**): [goldilocks-ebm.py](): Event-Based Model ([Fonteijn, et al., NeuroImage 2012](https://doi.org/10.1016/j.neuroimage.2012.01.062))

### Workflow for Z-Score SuStaIn

See [goldilocks-pysustain.py](./goldilocks-pysustain.py) for a worked example, but here's a conceptual overview.

* Import the module:

```
from goldilocks_dpm import goldilocks_ZscoreSustain
```

<br/>

* Prepare your data.
   - SuStaIn: see examples in [pySuStaIn](https://github.com/ucl-pond/pySuStaIn) for inspiration and guidance.
   - EBM: see [kde_ebm](https://github.com/ucl-pond/kde_ebm) and [disease-progression-modelling.github.io](https://disease-progression-modelling.github.io) for inspiration and guidance. 

<br/>

* Create a goldilocks ZScoreSustaIn object including your data matrix `X` and vector `y` labelling cases and controls (controls are used to z-score your data):

```
output_folder = Path.cwd() # or wherever you want the output to go

gdpm = goldilocks_ZscoreSustain(
    dpmData = X,
    classes = y,
    output_folder = output_folder,
    robust_zscores = False,
    case_label = 1,
    ctrl_label = 0, 
    direction_abnormal = direction_abnormal,
    biomarker_labels = biomarkers
)
```

<br/>

* Run the goldilocks zone calculation

```
gdpm.run_goldilocks(
    plot = True,
    plot_format = "png",
    verbose = False
)
```

<br/>

* Your `goldilocks_ZscoreSustain` object will now contain the z-scored X data, and goldilocks-suggested pySuStaIn hyperparameters `Z_vals` and `Z_max`

