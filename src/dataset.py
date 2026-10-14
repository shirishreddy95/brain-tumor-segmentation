"""BraTS 2020 dataset: loads 4 MRI modalities + segmentation, normalizes, crops/pads.

Extracted verbatim from MAU_Net_Brain_Tumor_Segmentation.ipynb — logic unchanged.
"""
import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, data_dir, crop_size=(96, 96, 96)):
        self.data_dir = data_dir
        self.crop_size = crop_size

        # Get patient directories
        self.patient_dirs = []
        if os.path.exists(os.path.join(data_dir, 'MICCAI_BraTS2020_TrainingData')):
            train_dir = os.path.join(data_dir, 'MICCAI_BraTS2020_TrainingData')
            self.patient_dirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir)
                               if os.path.isdir(os.path.join(train_dir, d)) and d.startswith('BraTS')]

        print(f"Found {len(self.patient_dirs)} patients")

        # Fixed: Files are .nii not .nii.gz
        self.modalities = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        self.seg_suffix = '_seg.nii'

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_id = os.path.basename(patient_dir)

        # Load modalities
        images = []
        for mod in self.modalities:
            path = os.path.join(patient_dir, patient_id + mod)
            if os.path.exists(path):
                img = nib.load(path).get_fdata()
                images.append(img)
            else:
                print(f"Missing: {path}")
                images.append(np.zeros((240, 240, 155)))

        # Load segmentation
        seg_path = os.path.join(patient_dir, patient_id + self.seg_suffix)
        seg = nib.load(seg_path).get_fdata() if os.path.exists(seg_path) else np.zeros_like(images[0])

        # Stack and convert
        image = torch.from_numpy(np.stack(images, axis=0)).float()
        seg = torch.from_numpy(seg).long()

        # Normalize each modality
        for i in range(4):
            img_data = image[i]
            if img_data.max() > 0:
                mean_val = img_data[img_data > 0].mean()
                std_val = img_data[img_data > 0].std()
                if std_val > 0:
                    image[i] = torch.clamp((img_data - mean_val) / std_val, -5, 5)

        # Crop/pad to target size
        image, seg = self._resize(image, seg)

        # Convert labels (4->3)
        seg[seg == 4] = 3

        return image, seg, patient_id

    def _resize(self, image, seg):
        current = image.shape[1:]
        target = self.crop_size

        for i in range(3):
            if current[i] > target[i]:
                start = (current[i] - target[i]) // 2
                end = start + target[i]
                if i == 0:
                    image, seg = image[:, start:end], seg[start:end]
                elif i == 1:
                    image, seg = image[:, :, start:end], seg[:, start:end]
                else:
                    image, seg = image[:, :, :, start:end], seg[:, :, start:end]
            else:
                pad = target[i] - current[i]
                pad_before, pad_after = pad // 2, pad - pad // 2
                if i == 0:
                    image = F.pad(image, (0, 0, 0, 0, pad_before, pad_after))
                    seg = F.pad(seg, (0, 0, 0, 0, pad_before, pad_after))
                elif i == 1:
                    image = F.pad(image, (0, 0, pad_before, pad_after))
                    seg = F.pad(seg, (0, 0, pad_before, pad_after))
                else:
                    image = F.pad(image, (pad_before, pad_after))
                    seg = F.pad(seg, (pad_before, pad_after))

        return image, seg
