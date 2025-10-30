import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import HipMRIDataset, make_loaders
from modules import Improved2DUNet, DiceLoss, dice_coefficient

# -----------------------------
# One epoch of training
# -----------------------------
def train_epoch(model, train_loader, criterion, optimizer, device, num_classes=6):
    model.train()
    running_loss = 0.0
    dice_scores_per_class = [[] for _ in range(num_classes)]

    pbar = tqdm(train_loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            dice_scores = dice_coefficient(outputs, masks, num_classes=num_classes)
            for i, score in enumerate(dice_scores):
                dice_scores_per_class[i].append(score)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice_c3': f'{dice_scores[3]:.4f}' if len(dice_scores) > 3 else 'n/a'
        })

    epoch_loss = running_loss / max(1, len(train_loader))
    avg_dice_per_class = [float(np.mean(scores)) if len(scores) else 0.0
                          for scores in dice_scores_per_class]
    return epoch_loss, avg_dice_per_class

# -----------------------------
# Validation
# -----------------------------
@torch.no_grad()
def validate(model, val_loader, criterion, device, num_classes=6):
    model.eval()
    running_loss = 0.0
    dice_scores_per_class = [[] for _ in range(num_classes)]

    pbar = tqdm(val_loader, desc='Validation')
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item()

        dice_scores = dice_coefficient(outputs, masks, num_classes=num_classes)
        for i, score in enumerate(dice_scores):
            dice_scores_per_class[i].append(score)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice_c3': f'{dice_scores[3]:.4f}' if len(dice_scores) > 3 else 'n/a'
        })

    epoch_loss = running_loss / max(1, len(val_loader))
    avg_dice_per_class = [float(np.mean(scores)) if len(scores) else 0.0
                          for scores in dice_scores_per_class]
    return epoch_loss, avg_dice_per_class

# -----------------------------
# Plot metrics (loss + min Dice across all classes)
# -----------------------------
def plot_metrics(train_losses, val_losses, train_dice, val_dice, save_path='training_metrics.png'):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss'); axes[0].legend(); axes[0].grid(True)

    train_dice_min = [min(d) for d in train_dice]
    val_dice_min   = [min(d) for d in val_dice]
    axes[1].plot(epochs, train_dice_min, 'b-', label='Train Dice (Min)')
    axes[1].plot(epochs, val_dice_min,   'r-', label='Val Dice (Min)')
    axes[1].axhline(y=0.75, color='g', linestyle='--', label='Target (0.75)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Minimum Dice Score (All Classes)'); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Metrics plot saved to {save_path}")

# -----------------------------
# Main training loop
# -----------------------------
def train_model(
    data_path,
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-4,
    num_classes=6,
    save_dir='checkpoints',
    resize=(256,128),
    num_workers=4
):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # === Dataloaders ===
    print("Loading datasets...")
    train_loader, val_loader, test_loader = make_loaders(
        data_root=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        resize=resize,
        normalize=True,
        num_classes=num_classes
    )

    # === Model ===
    print("Initializing model...")
    model = Improved2DUNet(in_channels=1, n_classes=num_classes, base=32, p_drop=0.2).to(device)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # === Loss & Optimizer ===
    criterion = DiceLoss()  # your Dice loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Simple plateau scheduler tracking min class dice (conservative)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # === Logs ===
    train_losses, val_losses = [], []
    train_dice_hist, val_dice_hist = [], []
    best_val_min_dice = 0.0

    print(f"\nStarting training for {num_epochs} epochs...\n")
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 50)

        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        val_loss,   val_dice   = validate(model, val_loader, criterion, device, num_classes)

        # Record
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_dice_hist.append(train_dice); val_dice_hist.append(val_dice)

        # Scheduler on min dice across classes
        min_val_dice = min(val_dice)
        scheduler.step(min_val_dice)

        # Print summary
        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("Train Dice: " + " ".join([f"C{i}:{d:.4f}" for i,d in enumerate(train_dice)]))
        print("Val   Dice: " + " ".join([f"C{i}:{d:.4f}" for i,d in enumerate(val_dice)]))

        # Save best by min class dice
        if min_val_dice > best_val_min_dice:
            best_val_min_dice = min_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved best model (min Dice across classes): {best_val_min_dice:.4f}")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

        print()

    # Plot curves
    plot_metrics(train_losses, val_losses, train_dice_hist, val_dice_hist,
                 save_path=os.path.join(save_dir, 'training_metrics.png'))

    print("\nTraining completed!")
    print(f"Best validation min Dice (all classes): {best_val_min_dice:.4f}")
    return model, train_losses, val_losses, train_dice_hist, val_dice_hist

# -----------------------------
# Example entry point
# -----------------------------
if __name__ == "__main__":
    data_path = "/content/drive/My Drive/keras_slices_data"
    num_epochs = 30
    batch_size = 8
    learning_rate = 1e-4

    model, train_losses, val_losses, train_dice, val_dice = train_model(
        data_path=data_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir='checkpoints',
        num_classes=6,
        resize=(256,128),
        num_workers=2
    )