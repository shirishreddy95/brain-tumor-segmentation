# BraTS 2020 Dataset Setup

The BraTS 2020 dataset is **not** included in this repository (gated medical imaging data).

## Download

1. Register at the [official BraTS 2020 page](https://www.med.upenn.edu/cbica/brats2020/) (CBICA, University of Pennsylvania).
2. Download `BraTS2020_TrainingData` and place it under this `data/` folder.

## Expected layout

```
data/
└── BraTS2020_TrainingData/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_flair.nii.gz
    │   ├── BraTS20_Training_001_t1.nii.gz
    │   ├── BraTS20_Training_001_t1ce.nii.gz
    │   ├── BraTS20_Training_001_t2.nii.gz
    │   └── BraTS20_Training_001_seg.nii.gz
    ├── BraTS20_Training_002/
    │   └── ...
    └── ...
```

The notebook's default `data_dir` is `./data/BraTS2020_TrainingData` — update it in the configuration cell if your data lives elsewhere.
