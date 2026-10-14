"""Visualization utilities: prediction overlays, architecture diagram, structure analysis.

Extracted verbatim from MAU_Net_Brain_Tumor_Segmentation.ipynb — logic unchanged.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SimpleMAUNet


def visualize_predictions(model, val_loader, device, num_samples=5):
    model.eval()

    # Get samples from dataset directly (before normalization)
    dataset = val_loader.dataset.dataset  # Get original dataset
    indices = list(range(len(dataset)))
    import random
    random_indices = random.sample(indices, min(num_samples, len(indices)))

    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))  # Added column for legend
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    class_names = ['Background', 'Necrotic', 'Edema', 'Enhancing']
    # Get actual colors from tab10 colormap
    import matplotlib.cm as cm
    tab10_colors = cm.get_cmap('tab10')
    class_colors = [tab10_colors(i) for i in range(4)]

    with torch.no_grad():
        for idx, data_idx in enumerate(random_indices):
            # Get original data
            images, targets, patient_id = dataset[data_idx]

            # Get prediction
            images_batch = images.unsqueeze(0).to(device)
            targets_batch = targets.unsqueeze(0).to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(images_batch)
            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            # Get middle slice
            mid_slice = images.shape[3] // 2  # Depth dimension

            # Get raw FLAIR (before normalization)
            img_slice = images[0, :, :, mid_slice].cpu().numpy()  # FLAIR channel
            target_slice = targets[:, :, mid_slice].cpu().numpy()
            pred_slice = pred[0, :, :, mid_slice].cpu().numpy()

            # Clip extreme values and normalize
            img_slice = np.clip(img_slice, np.percentile(img_slice, 1), np.percentile(img_slice, 99))
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

            # Calculate class statistics for the slice
            total_pixels = target_slice.size

            # Ground truth class counts
            gt_counts = {}
            pred_counts = {}

            for class_idx in range(4):
                gt_count = (target_slice == class_idx).sum()
                pred_count = (pred_slice == class_idx).sum()
                gt_counts[class_idx] = (gt_count, gt_count/total_pixels*100)
                pred_counts[class_idx] = (pred_count, pred_count/total_pixels*100)

            # Plot
            axes[idx, 0].imshow(img_slice, cmap='gray')
            axes[idx, 0].set_title(f'FLAIR\n{patient_id}')
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(target_slice, cmap='tab10', vmin=0, vmax=3)
            gt_text = "Ground Truth\n" + "\n".join([f"{class_names[i]}: {gt_counts[i][0]} ({gt_counts[i][1]:.1f}%)"
                                                   for i in range(4) if gt_counts[i][0] > 0])
            axes[idx, 1].set_title(gt_text, fontsize=9)
            axes[idx, 1].axis('off')

            axes[idx, 2].imshow(pred_slice, cmap='tab10', vmin=0, vmax=3)
            pred_text = "Prediction\n" + "\n".join([f"{class_names[i]}: {pred_counts[i][0]} ({pred_counts[i][1]:.1f}%)"
                                                    for i in range(4) if pred_counts[i][0] > 0])
            axes[idx, 2].set_title(pred_text, fontsize=9)
            axes[idx, 2].axis('off')

            # Overlay
            axes[idx, 3].imshow(img_slice, cmap='gray')
            mask = pred_slice > 0
            axes[idx, 3].imshow(np.ma.masked_where(~mask, pred_slice), cmap='Reds', alpha=0.5)
            axes[idx, 3].set_title('Overlay')
            axes[idx, 3].axis('off')

            # Legend (only for first row)
            if idx == 0:
                axes[idx, 4].axis('off')
                axes[idx, 4].set_title('Class Legend', fontsize=12, weight='bold')

                # Create legend patches
                for class_idx in range(4):
                    color = class_colors[class_idx]
                    # Create colored rectangle
                    rect = plt.Rectangle((0.1, 0.8 - class_idx * 0.2), 0.15, 0.1,
                                       facecolor=color, edgecolor='black')
                    axes[idx, 4].add_patch(rect)
                    # Add text
                    axes[idx, 4].text(0.3, 0.85 - class_idx * 0.2,
                                     f"Class {class_idx}: {class_names[class_idx]}",
                                     fontsize=10, va='center')

                axes[idx, 4].set_xlim(0, 1)
                axes[idx, 4].set_ylim(0, 1)
            else:
                # For other rows, hide the legend column
                axes[idx, 4].axis('off')

            # Print detailed stats (dice scores only in console)
            dice_scores = dice_coefficient(outputs, targets_batch)
            print(f"\n{patient_id}:")
            print(f"  Dice Scores - ET: {dice_scores[0]:.3f}, WT: {dice_scores[1]:.3f}, TC: {dice_scores[2]:.3f}")
            print(f"  Ground Truth Classes: {[f'{class_names[i]}: {gt_counts[i][1]:.1f}%' for i in range(4) if gt_counts[i][0] > 0]}")
            print(f"  Predicted Classes: {[f'{class_names[i]}: {pred_counts[i][1]:.1f}%' for i in range(4) if pred_counts[i][0] > 0]}")

    plt.tight_layout()
    plt.savefig('predictions_with_legend.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_network_architecture():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch, ConnectionPatch

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Define colors
    colors = {
        'input': '#E8F4FD',
        'encoder': '#B3D9FF',
        'bottleneck': '#FF6B6B',
        'decoder': '#98FB98',
        'output': '#FFD700',
        'skip': '#DDA0DD'
    }

    # Helper function to draw blocks
    def draw_block(x, y, width, height, text, color, fontsize=10):
        box = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.1",
                           facecolor=color,
                           edgecolor='black',
                           linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center', fontsize=fontsize, weight='bold')

    # Helper function to draw arrows
    def draw_arrow(x1, y1, x2, y2, color='black', style='->', width=2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle=style, color=color, lw=width))

    # Input
    draw_block(1, 6, 2, 1, 'Input\n4×96³', colors['input'], 9)

    # Encoder path
    encoder_blocks = [
        ('Conv+MDConv\n32 channels', 4, 8),
        ('Conv+MDConv\n64 channels', 4, 6.5),
        ('Conv+MDConv\n128 channels', 4, 5),
        ('Conv+MDConv\n256 channels', 4, 3.5)
    ]

    for i, (text, x, y) in enumerate(encoder_blocks):
        draw_block(x, y, 2.5, 1, text, colors['encoder'], 8)
        if i == 0:
            draw_arrow(3, 6.5, 4, 8.5)  # Input to first encoder
        if i < len(encoder_blocks) - 1:
            draw_arrow(5.25, y + 0.5, 5.25, encoder_blocks[i+1][2] + 1)  # Down arrows
            ax.text(5.5, y - 0.3, 'MaxPool', fontsize=7, style='italic')

    # Bottleneck
    draw_block(4, 2, 2.5, 1, 'Bottleneck\n512 channels', colors['bottleneck'], 8)
    draw_arrow(5.25, 3.5, 5.25, 3)

    # Decoder path
    decoder_blocks = [
        ('Conv+MDConv\n256 channels', 9, 3.5),
        ('Conv+MDConv\n128 channels', 9, 5),
        ('Conv+MDConv\n64 channels', 9, 6.5),
        ('Conv+MDConv\n32 channels', 9, 8)
    ]

    for i, (text, x, y) in enumerate(decoder_blocks):
        draw_block(x, y, 2.5, 1, text, colors['decoder'], 8)
        if i == 0:
            draw_arrow(6.5, 2.5, 9, 4)  # Bottleneck to first decoder
        if i < len(decoder_blocks) - 1:
            draw_arrow(10.25, y + 1, 10.25, decoder_blocks[i+1][2])  # Up arrows
            ax.text(10.5, y + 1.3, 'Upsample', fontsize=7, style='italic')

    # Skip connections
    skip_connections = [
        (6.5, 8.5, 9, 8.5),    # enc1 to dec4
        (6.5, 7, 9, 7),        # enc2 to dec3
        (6.5, 5.5, 9, 5.5),    # enc3 to dec2
        (6.5, 4, 9, 4)         # enc4 to dec1
    ]

    for x1, y1, x2, y2 in skip_connections:
        draw_arrow(x1, y1, x2, y2, color='purple', width=1.5)

    # Add skip connection label
    ax.text(7.75, 9, 'Skip Connections', ha='center', fontsize=9,
           color='purple', weight='bold')

    # Output
    draw_block(12.5, 8, 2, 1, 'Output\n4 classes', colors['output'], 9)
    draw_arrow(11.5, 8.5, 12.5, 8.5)

    # Add title and labels
    ax.text(8, 11, 'MAU-Net Architecture', ha='center', fontsize=16, weight='bold')
    ax.text(8, 10.5, 'MRI Brain Tumor Segmentation Network', ha='center', fontsize=12)

    # Add legend
    legend_elements = [
        ('Input Layer', colors['input']),
        ('Encoder Blocks', colors['encoder']),
        ('Bottleneck', colors['bottleneck']),
        ('Decoder Blocks', colors['decoder']),
        ('Output Layer', colors['output'])
    ]

    for i, (label, color) in enumerate(legend_elements):
        y_pos = 1.5 - i * 0.3
        draw_block(13, y_pos, 0.3, 0.2, '', color)
        ax.text(13.5, y_pos + 0.1, label, fontsize=9, va='center')

    # Add model info
    info_text = f"""Model Information:
• Input: 4 MRI modalities (FLAIR, T1, T1ce, T2)
• Architecture: U-Net with Mixed Depth-wise Convolutions
• Output: 4-class segmentation (Background, Necrotic, Edema, Enhancing)
• Parameters: ~2.8M
• Optimized for: RTX 2060 8GB"""

    ax.text(0.5, 4, info_text, fontsize=9, va='top',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig('mau_net_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Network architecture saved as 'mau_net_architecture.png'")


def analyze_network_structure():
    model = SimpleMAUNet()

    print("=== MAU-Net Architecture Analysis ===\n")

    # Input Layer
    print("INPUT LAYER:")
    print(f"• Nodes: 4 × 96³ = {4 * 96**3:,} input features")
    print(f"• Represents: 4 MRI modalities (FLAIR, T1, T1ce, T2)")

    print("\nHIDDEN LAYERS:")

    # Count layers and parameters
    layer_count = 0
    total_params = 0

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv3d, torch.nn.BatchNorm3d)):
            layer_count += 1
            params = sum(p.numel() for p in module.parameters())
            total_params += params

            if isinstance(module, torch.nn.Conv3d):
                in_ch, out_ch = module.in_channels, module.out_channels
                kernel = module.kernel_size
                print(f"• Layer {layer_count}: Conv3d({in_ch}→{out_ch}, kernel={kernel}) - {params:,} params")

    print(f"\nTOTAL HIDDEN LAYERS: {layer_count}")

    # Feature map sizes through network
    print(f"\nFEATURE MAP PROGRESSION:")
    x = torch.randn(1, 4, 96, 96, 96)

    # Track through encoder
    sizes = []
    e1 = model.enc1(x)
    sizes.append(f"Encoder 1: {e1.shape[1]} channels × {e1.shape[2]}³ = {e1.numel():,} features")

    e2 = model.enc2(torch.nn.functional.max_pool3d(e1, 2))
    sizes.append(f"Encoder 2: {e2.shape[1]} channels × {e2.shape[2]}³ = {e2.numel():,} features")

    e3 = model.enc3(torch.nn.functional.max_pool3d(e2, 2))
    sizes.append(f"Encoder 3: {e3.shape[1]} channels × {e3.shape[2]}³ = {e3.numel():,} features")

    e4 = model.enc4(torch.nn.functional.max_pool3d(e3, 2))
    sizes.append(f"Encoder 4: {e4.shape[1]} channels × {e4.shape[2]}³ = {e4.numel():,} features")

    b = model.bottleneck(torch.nn.functional.max_pool3d(e4, 2))
    sizes.append(f"Bottleneck: {b.shape[1]} channels × {b.shape[2]}³ = {b.numel():,} features")

    for size in sizes:
        print(f"• {size}")

    print(f"\nOUTPUT LAYER:")
    print(f"• Nodes: 4 × 96³ = {4 * 96**3:,} output features")
    print(f"• Represents: 4 classes (Background, Necrotic, Edema, Enhancing)")

    print(f"\nSUMMARY:")
    print(f"• Total Parameters: {total_params:,}")
    print(f"• Total Layers: {layer_count}")
    print(f"• Input Features: {4 * 96**3:,}")
    print(f"• Output Features: {4 * 96**3:,}")
    print(f"• Architecture: Encoder-Decoder with Skip Connections")


if __name__ == '__main__':
    # Self-contained figures (no trained model needed):
    visualize_network_architecture()
    analyze_network_structure()
    # Prediction overlays need a trained model + val_loader, e.g.:
    # visualize_predictions(model, val_loader, device, num_samples=5)
