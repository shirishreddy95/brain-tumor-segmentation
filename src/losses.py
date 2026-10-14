"""Combined Dice + Cross-Entropy loss and the Dice coefficient metric.

Extracted verbatim from MAU_Net_Brain_Tumor_Segmentation.ipynb — logic unchanged.
"""
import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
    
    def dice_loss(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        dice_scores = []
        for i in range(pred.shape[1]):
            pred_i = pred[:, i].flatten()
            target_i = (target == i).float().flatten()
            intersection = (pred_i * target_i).sum()
            dice = (2 * intersection + 1e-6) / (pred_i.sum() + target_i.sum() + 1e-6)
            dice_scores.append(dice)
        return 1 - torch.stack(dice_scores).mean()
    
    def forward(self, pred, target):
        ce = self.ce(pred, target)
        dice = self.dice_loss(pred, target)
        return (1 - self.alpha) * ce + self.alpha * dice

def dice_coefficient(pred, target):
    pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    dice_scores = []
    for i in range(1, 4):  # Skip background
        pred_i = (pred == i).float()
        target_i = (target == i).float()
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()
        dice = (2 * intersection) / union if union > 0 else 1.0
        dice_scores.append(float(dice))  # Convert to float, remove .item()
    return dice_scores
