# Brain Tumor Segmentation with MAU-Net on BraTS 2020

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Dataset-BraTS%202020-green)](https://www.med.upenn.edu/cbica/brats2020/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

3D brain tumor segmentation from multimodal MRI using a **Multiscale Attention U-Net (MAU-Net)**, implemented in PyTorch following Chen et al. (2024). Built as the final project for **AI 700 (Deep Learning), Long Island University**.

![Sample predictions](assets/predictions_with_legend.jpg)

## Results

Final validation scores after 50 epochs (BraTS 2020 training split, 80/20):

| Region | Dice |
|---|---|
| Enhancing Tumor (ET) | 0.616 |
| Whole Tumor (WT) | 0.800 |
| Tumor Core (TC) | 0.834 |
| **Average** | **0.750** |

- Validation accuracy ~89%; final train loss 0.1088, validation loss 0.1329
- Train/val loss gap ≈ 0.02 — no overfitting
- Improved from 0.693 average Dice at midterm → 0.750 final

![Training curves](assets/training_results.jpg)

## Model

MAU-Net extends the classic 3D U-Net with multi-scale attention:

1. **MDConv (Mixed Depth-wise Convolution)** — multi-scale feature extraction in the encoder
2. **CPM (Context Pyramid Module)** — attention over global context
3. **Self-ensemble** — feature fusion across decoder levels

Input: 4 MRI modalities (FLAIR, T1, T1ce, T2) as 96³ volumes. See [model.md](model.md) for the full layer-by-layer architecture, and the [Citation](#citation) section for the original paper.

![Architecture](assets/mau_net_architecture.jpg)

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download BraTS 2020 (registration required) and place it under ./data
#    See data/README.md for instructions and the expected folder layout

# 4. Launch the notebook
jupyter notebook
```

Open `MAU_Net_Brain_Tumor_Segmentation.ipynb` and run the cells top to bottom. The notebook covers model definition, dataset loading, training with TensorBoard logging, evaluation, and prediction visualization.

Training configuration:

| Setting | Value |
|---|---|
| Optimizer | Adam, lr = 1e-3 |
| Loss | Combined Cross-Entropy + Dice (α = 0.25) |
| Batch size | 2 |
| Crop size | 96 × 96 × 96 |
| Epochs | 50 |
| Precision | Mixed (FP16) |

## Project structure

```
brain-tumor-segmentation/
├── MAU_Net_Brain_Tumor_Segmentation.ipynb  # Main notebook: model, training, evaluation
├── model.md                                # Layer-by-layer architecture reference
├── requirements.txt                        # Python dependencies
├── assets/                                 # Figures (architecture, predictions, curves)
└── data/                                   # BraTS 2020 goes here (not included, see below)
```

## Dataset

BraTS 2020 is **not** included in this repo (gated medical data). Download it from the [official BraTS 2020 page](https://www.med.upenn.edu/cbica/brats2020/) after registration, then follow [data/README.md](data/README.md) for the expected layout.

Segmentation labels: `0` background · `1` necrotic/non-enhancing core · `2` peritumoral edema · `4` enhancing tumor (mapped to 3 in the model). Evaluation regions: **WT** = {1,2,4}, **TC** = {1,4}, **ET** = {4}.

## Notes

- Developed on an RTX 2060 (8 GB): 96³ crops, batch size 2, and FP16 mixed precision keep training within memory limits. Reduce crop size if you hit OOM errors.
- ET is the hardest region (small, sparse enhancing regions) — the most promising next step is loss rebalancing and stronger augmentation targeted at ET.

## Citation

```bibtex
@article{chen2024mri,
  title   = {MRI Brain Tumour Segmentation Using Multiscale Attention U-Net},
  author  = {Chen, Bonian and He, Tao and Wang, Weizhuo and Han, Yutong and Zhang, Jianxin and Bobek, Samo and Zabukovsek, Simona Sternad},
  journal = {Informatica},
  volume  = {35},
  number  = {4},
  pages   = {751--774},
  year    = {2024}
}
```

## Author

**Shirish Kumar Reddy Kaithi** — MS Artificial Intelligence, Long Island University (2026)
