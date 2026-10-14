"""MAU-Net model: MDConv, AGCB, CPM blocks and the SimpleMAUNet 3D U-Net.

Extracted verbatim from MAU_Net_Brain_Tumor_Segmentation.ipynb — logic unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMAUNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, dropout_rate=0.2):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate * 2)
        )

        # Decoder
        self.dec4 = nn.Sequential(
            nn.Conv3d(512 + 256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        self.dec3 = nn.Sequential(
            nn.Conv3d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        self.dec2 = nn.Sequential(
            nn.Conv3d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        self.dec1 = nn.Sequential(
            nn.Conv3d(64 + 32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv3d(32, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        e3 = self.enc3(F.max_pool3d(e2, 2))
        e4 = self.enc4(F.max_pool3d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool3d(e4, 2))

        # Decoder
        d4 = F.interpolate(b, e4.shape[2:], mode='trilinear')
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = F.interpolate(d4, e3.shape[2:], mode='trilinear')
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, e2.shape[2:], mode='trilinear')
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, e1.shape[2:], mode='trilinear')
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)


if __name__ == '__main__':
    # Test the simplified model
    model = SimpleMAUNet()
    x = torch.randn(1, 4, 96, 96, 96)
    model.eval()
    with torch.no_grad():
        out = model(x)
        print(f"Input: {x.shape}, Output: {out.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
