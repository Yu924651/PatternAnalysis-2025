"""
Train script for Improved2DUNet on HipMRI dataset (simple FP32 version)
"""
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from modules import Improved2DUNet, DiceLoss
from dataset import make_loaders

# -----------------------------
# One epoch of training (FP32)
# -----------------------------
# -----------------------------
# One epoch of training (FP32) with streaming per-class Dice
# -----------------------------
def train_epoch(model, train_loader, criterion, optimizer, device, num_classes=6, eps=1e-6, report_class=3):
    """
    Runs one training epoch in FP32.
    Returns:
      - epoch loss (float)
      - per-class Dice for the whole epoch as a list of length C
    """
    model.train()
    running_loss = 0.0

    # Streaming accumulators for Dice: TP, P (pred positives), T (true positives)
    tp = torch.zeros(num_classes, device=device, dtype=torch.float64)
    p  = torch.zeros(num_classes, device=device, dtype=torch.float64)
    t  = torch.zeros(num_classes, device=device, dtype=torch.float64)

    pbar = tqdm(train_loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward → loss → backward → step
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # ---- streaming per-class Dice stats (no grad) ----
        with torch.no_grad():
            preds = outputs.argmax(dim=1)  # (B,H,W)

            # Update counts per class in a vectorized way
            for c in range(num_classes):
                pc = (preds == c)
                tc = (masks == c)
                tp[c] += (pc & tc).sum()
                p[c]  += pc.sum()
                t[c]  += tc.sum()

            # Quick live preview for one class (e.g., class 3)
            if 0 <= report_class < num_classes:
                dice_c = (2.0 * tp[report_class] + eps) / (p[report_class] + t[report_class] + eps)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    f'dice_c{report_class}': f'{float(dice_c):.4f}'
                })
            else:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Epoch-level Dice per class from accumulated counts
    dice_per_class = ((2.0 * tp + eps) / (p + t + eps)).tolist()
    epoch_loss = running_loss / max(1, len(train_loader))
    return epoch_loss, dice_per_class


# -----------------------------
# Validation (FP32, no grad)
# -----------------------------
@torch.no_grad()
def validate(model, val_loader, criterion, device, num_classes=6, eps=1e-6, report_class=3):
    """
    Validation loop with streaming per-class Dice (unbiased over the whole epoch).
    Returns:
      - epoch_loss (float)
      - dice_per_class (list length C)
    """
    model.eval()
    running_loss = 0.0

    # Streaming accumulators for Dice: TP, P (pred positives), T (true positives)
    tp = torch.zeros(num_classes, device=device, dtype=torch.float64)
    p  = torch.zeros(num_classes, device=device, dtype=torch.float64)
    t  = torch.zeros(num_classes, device=device, dtype=torch.float64)

    pbar = tqdm(val_loader, desc='Validation')
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        # Forward + loss (no backward on val)
        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item()

        # Update streaming Dice stats
        preds = outputs.argmax(dim=1)  # (B,H,W)
        for c in range(num_classes):
            pc = (preds == c)
            tc = (masks == c)
            tp[c] += (pc & tc).sum()
            p[c]  += pc.sum()
            t[c]  += tc.sum()

        # Live preview for a single class (e.g., class 3)
        if 0 <= report_class < num_classes:
            dice_c = (2.0 * tp[report_class] + eps) / (p[report_class] + t[report_class] + eps)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                f'dice_c{report_class}': f'{float(dice_c):.4f}'
            })
        else:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Epoch-level Dice per class
    dice_per_class = ((2.0 * tp + eps) / (p + t + eps)).tolist()
    epoch_loss = running_loss / max(1, len(val_loader))
    return epoch_loss, dice_per_class


# -----------------------------
# Main training loop (FP32)
# -----------------------------
def train_model(
    data_path,
    num_epochs=30,
    batch_size=8,
    learning_rate=1e-4,
    num_classes=6,
    save_dir='/content/drive/My Drive/checkpoints',
    resize=(256,128),
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataloaders
    train_loader, val_loader, test_loader = make_loaders(
        data_root=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        resize=resize,
        normalize=True,
        num_classes=num_classes,
    )

    # Model / loss / optimizer
    model = Improved2DUNet(in_channels=1, n_classes=num_classes, base=32, p_drop=0.2).to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_val_min_dice = 0.0  # we keep a single best-by-min-class-Dice metric

    print(f"\nStarting training for {num_epochs} epochs...\n")
    epoch_start = time.time()
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 50)

        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device,
                                             num_classes=num_classes)
        val_loss,   val_dice   = validate(model, val_loader, criterion, device,
                                          num_classes=num_classes)

        min_val_dice = min(val_dice)

        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("Train Dice: " + " ".join([f"C{i}:{d:.4f}" for i,d in enumerate(train_dice)]))
        print("Val   Dice: " + " ".join([f"C{i}:{d:.4f}" for i,d in enumerate(val_dice)]))

        # Save best by min class Dice
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

        # ETA
        epoch_time = time.time() - epoch_start
        remaining = num_epochs - epoch
        eta_min = (epoch_time * remaining) / 60.0
        print(f"Epoch time: {epoch_time:.1f}s | ETA: ~{eta_min:.1f} minutes\n")
        epoch_start = time.time()

    print("\nTraining completed!")
    print(f"Best validation min Dice (all classes): {best_val_min_dice:.4f}")
    return model


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
        save_dir='/content/drive/My Drive/checkpoints',
        num_classes=6,
        resize=(256,128),
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True
    )