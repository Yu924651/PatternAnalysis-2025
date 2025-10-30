import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset import make_loaders
from modules import Improved2DUNet, DiceLoss, dice_all_class

# --- SPEED FLAGS (must be set early) ---
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    # Allow fast TensorFloat-32 on Ampere+/Hopper GPUs
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    # PyTorch 2.x: improves matmul speed/precision policy
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")


# -----------------------------
# One epoch of training (AMP + grad clip)
# -----------------------------
def train_epoch(model, train_loader, criterion, optimizer, device, num_classes=6, amp=True, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    dice_scores_per_class = [[] for _ in range(num_classes)]

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    pbar = tqdm(train_loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            dice_scores = dice_all_class(outputs, masks, num_classes=num_classes)
            for i, score in enumerate(dice_scores):
                dice_scores_per_class[i].append(score)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice_socre': f'{dice_scores[3]:.4f}' if len(dice_scores) > 3 else 'n/a'
        })

    epoch_loss = running_loss / max(1, len(train_loader))
    avg_dice_per_class = [float(np.mean(scores)) if len(scores) else 0.0
                          for scores in dice_scores_per_class]
    return epoch_loss, avg_dice_per_class

# -----------------------------
# Validation (AMP for speed)
# -----------------------------
@torch.no_grad()
def validate(model, val_loader, criterion, device, num_classes=6, amp=True):
    model.eval()
    running_loss = 0.0
    dice_scores_per_class = [[] for _ in range(num_classes)]

    pbar = tqdm(val_loader, desc='Validation')
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)

        running_loss += loss.item()

        dice_scores = dice_all_class(outputs, masks, num_classes=num_classes)
        for i, score in enumerate(dice_scores):
            dice_scores_per_class[i].append(score)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice_score': f'{dice_scores[3]:.4f}' if len(dice_scores) > 3 else 'n/a'
        })

    epoch_loss = running_loss / max(1, len(val_loader))
    avg_dice_per_class = [float(np.mean(scores)) if len(scores) else 0.0
                          for scores in dice_scores_per_class]
    return epoch_loss, avg_dice_per_class

# -----------------------------
# Main training loop (AMP + ETA + tuned loaders)
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
    amp=True,                   # turn AMP on
    grad_clip=1.0,              # small clip stabilizes AMP
    prefetch_factor=2,          # faster input pipeline
    persistent_workers=True     # workers stay alive between epochs
):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # === Dataloaders ===
    print("Loading datasets...")
    # If you control make_loaders, pass through num_workers; otherwise, adjust inside it.
    train_loader, val_loader, test_loader = make_loaders(
        data_root=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        resize=resize,
        normalize=True,
        num_classes=num_classes,
    )

    # Try to set loader-level speed knobs if these attributes exist
    # (depends on how make_loaders builds DataLoaders)
    for loader in [train_loader, val_loader, test_loader]:
        if hasattr(loader, "prefetch_factor"):
            try: loader.prefetch_factor = prefetch_factor
            except Exception: pass
        if hasattr(loader, "persistent_workers"):
            try: loader.persistent_workers = (persistent_workers and num_workers > 0)
            except Exception: pass

    # === Model ===
    print("Initializing model...")
    model = Improved2DUNet(in_channels=1, n_classes=num_classes, base=32, p_drop=0.2).to(device)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # === Loss & Optimizer ===
    criterion = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Plateau scheduler on min class dice (no verbose arg for compatibility)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

    # === Logs ===
    train_losses, val_losses = [], []
    train_dice_hist, val_dice_hist = [], []
    best_val_min_dice = 0.0

    print(f"\nStarting training for {num_epochs} epochs...\n")
    epoch_start = time.time()
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 50)

        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device,
                                             num_classes=num_classes, amp=amp, grad_clip=grad_clip)
        val_loss,   val_dice   = validate(model, val_loader, criterion, device,
                                          num_classes=num_classes, amp=amp)

        # Record
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_dice_hist.append(train_dice); val_dice_hist.append(val_dice)

        # Scheduler on min dice across classes
        min_val_dice = min(val_dice)
        scheduler.step(min_val_dice)

        # Print summary
        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
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

        # --- Accurate ETA ---
        epoch_time = time.time() - epoch_start
        remaining = num_epochs - epoch
        eta_min = (epoch_time * remaining) / 60.0
        print(f"Epoch time: {epoch_time:.1f}s | ETA: ~{eta_min:.1f} minutes\n")
        epoch_start = time.time()


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
        save_dir='/content/drive/My Drive/checkpoints',
        num_classes=6,
        resize=(256,128),
        num_workers=4,         
        amp=True,               
        grad_clip=1.0,
        prefetch_factor=2,
        persistent_workers=True
    )