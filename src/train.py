"""Training script: config, data loaders, TensorBoard logging, train/val loop.

Extracted verbatim from MAU_Net_Brain_Tumor_Segmentation.ipynb — logic unchanged.
"""
import datetime
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import BraTSDataset
from losses import CombinedLoss, dice_coefficient
from model import SimpleMAUNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # Configuration
    config = {
        'data_dir': './data/BraTS2020_TrainingData',
        'batch_size': 2,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'crop_size': (96, 96, 96)
    }

    # Create dataset
    dataset = BraTSDataset(config['data_dir'], config['crop_size'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Initialize model, loss, optimizer
    model = SimpleMAUNet(dropout_rate=0.2).to(device)
    criterion = CombinedLoss(alpha=0.25)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    scaler = GradScaler('cuda')  # Fixed deprecated warning

    # Training history
    train_losses, val_losses, val_dice_scores = [], [], []

    print("✅ Setup complete - ready to train!")
    # Add TensorBoard logging
    from torch.utils.tensorboard import SummaryWriter
    import datetime

    # Create TensorBoard writer
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/MAU_Net_{timestamp}')

    # Log model architecture (run once)
    sample_input = torch.randn(1, 4, 96, 96, 96).to(device)
    writer.add_graph(model, sample_input)

    print("✅ TensorBoard setup complete!")
    print("Run: tensorboard --logdir=runs")
    best_val_loss = float('inf')

    # Modified training functions with logging
    best_val_loss = float('inf')

    # Modified training functions with logging
    def train_epoch_with_logging(epoch):
        model.train()
        total_loss = 0

        for batch_idx, (images, targets, _) in enumerate(tqdm(train_loader, desc='Training')):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Log every 50 batches
            if batch_idx % 50 == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/Train_Batch', loss.item(), step)

            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
        return avg_loss

    def validate_with_logging(epoch):
        model.eval()
        total_loss = 0
        all_dice = []

        with torch.no_grad():
            for images, targets, _ in tqdm(val_loader, desc='Validation'):
                images, targets = images.to(device), targets.to(device)

                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                total_loss += loss.item()
                dice = dice_coefficient(outputs, targets)
                all_dice.append(dice)

        avg_loss = total_loss / len(val_loader)
        avg_dice = np.mean(all_dice, axis=0)

        # Log validation metrics
        writer.add_scalar('Loss/Validation', avg_loss, epoch)
        writer.add_scalar('Dice/ET', avg_dice[0], epoch)
        writer.add_scalar('Dice/WT', avg_dice[1], epoch)
        writer.add_scalar('Dice/TC', avg_dice[2], epoch)
        writer.add_scalar('Dice/Average', np.mean(avg_dice), epoch)

        # Log learning rate
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        return avg_loss, avg_dice

    # Enhanced main training loop
    for epoch in range(config['num_epochs']):
        print(f'\nEpoch {epoch+1}/{config["num_epochs"]}')

        # Train and validate with logging
        train_loss = train_epoch_with_logging(epoch)
        val_loss, val_dice = validate_with_logging(epoch)

        # Update history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)

        scheduler.step(val_loss)

        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Dice - ET: {val_dice[0]:.3f}, WT: {val_dice[1]:.3f}, TC: {val_dice[2]:.3f}')

        # Log sample predictions every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_images, sample_targets, _ = next(iter(val_loader))
                sample_images = sample_images.to(device)
                sample_outputs = model(sample_images)
                sample_pred = torch.argmax(torch.softmax(sample_outputs, dim=1), dim=1)

                # Log images (middle slice)
                mid_slice = sample_images.shape[4] // 2
                writer.add_image(f'Input/FLAIR_Epoch_{epoch+1}',
                               sample_images[0, 0, :, :, mid_slice], epoch, dataformats='HW')
                writer.add_image(f'Target/Ground_Truth_Epoch_{epoch+1}',
                               sample_targets[0, :, :, mid_slice], epoch, dataformats='HW')
                writer.add_image(f'Prediction/Model_Output_Epoch_{epoch+1}',
                               sample_pred[0, :, :, mid_slice], epoch, dataformats='HW')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_mau_net.pth')
            print('✓ Best model saved!')

    # Close writer
    writer.close()
    print("\n✅ Training complete! Check TensorBoard: tensorboard --logdir=runs")


if __name__ == '__main__':
    main()
