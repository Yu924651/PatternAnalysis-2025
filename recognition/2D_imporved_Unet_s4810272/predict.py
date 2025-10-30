"""
Fast 2D U-Net Trainer (Mixed Precision + channels_last)

Compatible with:
- Improved2DUNet, DiceLoss, dice_per_class_from_logits in modules.py
- HipMRIDataset / make_loaders in dataset.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ---- bring your components ----
from dataset import make_loaders
from modules import Improved2DUNet, DiceLoss, dice_per_class_from_logits

# ================= CONFIG =================
DATA_ROOT = "/content/drive/My Drive/keras_slices_data"
OUTDIR     = "/content/drive/My Drive"
IN_CHANNELS = 1
NUM_CLASSES = 6
PROSTATE_CLASS = 1
BASE = 32
RESIZE = (256, 128)

EPOCHS = 30
BATCH_SIZE = 8       
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 8
GRAD_CLIP = 1.0
PATIENCE = 5
# ==========================================

def format_dice_row(dice_vec, highlight_idx=None, prefix=""):
    """Return a string like: C0:0.9123 C1:[0.8450] C2:0.1234 ..."""
    parts = []
    for i, d in enumerate(dice_vec):
        if highlight_idx is not None and i == highlight_idx:
            parts.append(f"C{i}:[{d:.4f}]")
        else:
            parts.append(f"C{i}:{d:.4f}")
    return (prefix + " ".join(parts)).strip()


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    has_cuda = (device.type == "cuda")

    sum_loss = 0.0
    steps = 0
    dice_sum = np.zeros(NUM_CLASSES, dtype=np.float64)

    pbar = tqdm(loader, desc="TRAIN", ncols=120)
    for x, y in pbar:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # AMP for speed
        with torch.amp.autocast("cuda", enabled=has_cuda):
            logits = model(x)
            loss = 0.5 * loss_fn(logits, y) + 0.5 * nn.functional.cross_entropy(logits, y)

        loss.backward()
        if GRAD_CLIP and GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        steps += 1
        sum_loss += float(loss)
        dice_sum += dice_per_class_from_logits(logits, y, NUM_CLASSES)

        avg_loss = sum_loss / steps
        avg_dice = dice_sum / steps
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "macro": f"{avg_dice.mean():.4f}",
            "pros": f"{avg_dice[PROSTATE_CLASS]:.4f}",
        })

    return (sum_loss / steps), (dice_sum / steps)


@torch.no_grad()
def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    has_cuda = (device.type == "cuda")

    sum_loss = 0.0
    steps = 0
    dice_sum = np.zeros(NUM_CLASSES, dtype=np.float64)

    pbar = tqdm(loader, desc="VAL", ncols=120, colour="cyan")
    for x, y in pbar:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=has_cuda):
            logits = model(x)
            loss = 0.5 * loss_fn(logits, y) + 0.5 * nn.functional.cross_entropy(logits, y)

        steps += 1
        sum_loss += float(loss)
        dice_sum += dice_per_class_from_logits(logits, y, NUM_CLASSES)

        avg_loss = sum_loss / steps
        avg_dice = dice_sum / steps
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "macro": f"{avg_dice.mean():.4f}",
            "pros": f"{avg_dice[PROSTATE_CLASS]:.4f}",
        })

    return (sum_loss / steps), (dice_sum / steps)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Data ----
    os.makedirs(OUTDIR, exist_ok=True)
    train_loader, val_loader, test_loader = make_loaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        resize=RESIZE,
        normalize=True,
        num_classes=NUM_CLASSES,
        transform_train=None,
        transform_eval=None,
    )
    # optional: persistent workers for a tiny speed bump
    for ld in (train_loader, val_loader, test_loader):
        ld.persistent_workers = True

    # ---- Model / Optim ----
    model = Improved2DUNet(IN_CHANNELS, NUM_CLASSES, BASE)
    model = model.to(device).to(memory_format=torch.channels_last)

    loss_fn = DiceLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_pros = 0.0
    no_improve = 0

    print("\nStarting training...\n")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n================= EPOCH {epoch}/{EPOCHS} =================")

        tr_loss, tr_dice = train_epoch(model, train_loader, optimizer, loss_fn, device)
        va_loss, va_dice = validate_epoch(model, val_loader, loss_fn, device)

        # ---- Pretty printing (includes per-class) ----
        print(f"TRAIN: loss={tr_loss:.4f} | macro={tr_dice.mean():.4f}")
        print("   " + format_dice_row(tr_dice, highlight_idx=PROSTATE_CLASS, prefix="Train Dice → "))

        print(f"VAL:   loss={va_loss:.4f} | macro={va_dice.mean():.4f}")
        print("   " + format_dice_row(va_dice, highlight_idx=PROSTATE_CLASS, prefix="Val   Dice → "))

        # ---- Checkpointing ----
        torch.save(model.state_dict(), os.path.join(OUTDIR, "last.pt"))

        if va_dice[PROSTATE_CLASS] > best_pros:
            best_pros = float(va_dice[PROSTATE_CLASS])
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(OUTDIR, "best.pt"))
            print(f"Best updated! Val prostate Dice = {best_pros:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping — no validation improvement.")
                break

    print("\nTraining finished.")

    # ---- Test using best weights ----
    if os.path.isfile(os.path.join(OUTDIR, "best.pt")):
        model.load_state_dict(torch.load(os.path.join(OUTDIR, "best.pt"), map_location=device))
        print("Loaded best.pt for testing")
    else:
        print("best.pt not found; using last.pt")
        model.load_state_dict(torch.load(os.path.join(OUTDIR, "last.pt"), map_location=device))

    model.eval()
    with torch.no_grad():
        sum_dice = np.zeros(NUM_CLASSES, dtype=np.float64)
        steps = 0
        for x, y in tqdm(test_loader, desc="TEST", ncols=120, colour="green"):
            x = x.to(device).to(memory_format=torch.channels_last)
            y = y.to(device)
            logits = model(x)
            sum_dice += dice_per_class_from_logits(logits, y, NUM_CLASSES)
            steps += 1

        test_dice = sum_dice / max(1, steps)
        print("\n===== TEST RESULTS =====")
        print(" " + format_dice_row(test_dice, highlight_idx=PROSTATE_CLASS, prefix="Per-class → "))
        print(f" Macro Dice: {test_dice.mean():.4f}")
        print(f" Prostate (C{PROSTATE_CLASS}) Dice: {test_dice[PROSTATE_CLASS]:.4f}")


if __name__ == "__main__":
    main()
