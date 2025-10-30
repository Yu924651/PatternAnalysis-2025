# === Predict + Visualize for HipMRI (Colab-ready) ===
# - Load trained Improved2DUNet from checkpoint
# - Predict one file OR a whole folder of .nii.gz 2D slices
# - Save masks and visualize input/mask/overlay

import os
from glob import glob
from typing import Optional, Tuple, List

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from modules import Improved2DUNet



# --------------------------
# Config (edit as needed)
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
RESIZE: Optional[Tuple[int, int]] = (256, 128)  # must match training
CHECKPOINT_PATH = "/content/drive/My Drive/checkpoints/best_model.pth"

# Example paths
TEST_IMAGE  = "/content/drive/My Drive/keras_slices_data/keras_slices_test/case_040_week_0_slice_0.nii.gz"
SAVE_SINGLE = "/content/drive/My Drive/predictions/save_output"

TEST_FOLDER = "/content/drive/My Drive/keras_slices_data/keras_slices_test"
SAVE_FOLDER = "/content/drive/My Drive/save_output"


# --------------------------
# Helpers
# --------------------------
def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = float(x.mean())
    sd = float(x.std())
    return (x - mu) / (sd + eps)

def load_model(checkpoint_path: str, n_classes: int = NUM_CLASSES) -> torch.nn.Module:
    model = Improved2DUNet(in_channels=1, n_classes=n_classes, base=32, p_drop=0.2)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model

@torch.no_grad()
def predict_tensor(model: torch.nn.Module, img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: (1,1,H,W) float32 normalized, already resized if needed
    returns np.ndarray mask (H,W) with class ids [0..NUM_CLASSES-1]
    """
    use_amp = (DEVICE.type == "cuda")
    with torch.cuda.amp.autocast(enabled=use_amp):
        logits = model(img_t)
        pred = torch.argmax(logits, dim=1)  # (1,H,W)
    return pred.squeeze(0).detach().cpu().numpy()

@torch.no_grad()
def predict_one(model: torch.nn.Module,
                image_path: str,
                save_path: Optional[str] = None,
                resize: Optional[Tuple[int, int]] = RESIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (image_2d_float, pred_mask_int) and optionally saves mask as NIfTI.
    """
    # --- Load NIfTI ---
    nii = nib.load(image_path)
    img = nii.get_fdata(dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D NIfTI but got shape {img.shape} at {image_path}")

    # --- Preprocess ---
    img_norm = zscore(img)
    img_t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,W)

    # Keep original affine for saving out
    affine = nii.affine

    # Optional resize to training size
    if resize is not None and img_t.shape[2:] != resize:
        img_t = F.interpolate(img_t, size=resize, mode="bilinear", align_corners=False)

    # --- Inference ---
    pred = predict_tensor(model, img_t)  # (H_resized, W_resized)

    # Save using the same affine as input (note: shape may differ if resized)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # If we resized, we may want to up/downsample back to original size.
        if resize is not None and pred.shape != img.shape:
            pred_back = torch.from_numpy(pred[None, None].astype(np.float32))
            pred_back = F.interpolate(pred_back, size=img.shape, mode="nearest")
            pred_back = pred_back.squeeze().numpy().astype(np.uint8)
        else:
            pred_back = pred.astype(np.uint8)

        nib.save(nib.Nifti1Image(pred_back, affine), save_path)
        print(f"Saved predicted mask → {save_path}")

    return img_norm, pred  # return normalized image (for viz) and resized pred

def visualize_triplet(img2d: np.ndarray, mask2d: np.ndarray, alpha: float = 0.35, title: str = ""):
    """
    Show input, mask, and overlay side-by-side.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img2d, cmap="gray")
    plt.title("Input MRI")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask2d, cmap="nipy_spectral", interpolation="nearest", vmin=0, vmax=NUM_CLASSES-1)
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img2d, cmap="gray")
    plt.imshow(mask2d, cmap="nipy_spectral", interpolation="nearest", alpha=alpha, vmin=0, vmax=NUM_CLASSES-1)
    plt.title("Overlay")
    plt.axis("off")

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def predict_folder(model: torch.nn.Module,
                   image_dir: str,
                   save_dir: Optional[str] = None,
                   max_visualize: int = 4,
                   resize: Optional[Tuple[int, int]] = RESIZE) -> List[str]:
    """
    Predicts all .nii.gz in a folder. Optionally saves masks, and visualizes a few.
    Returns list of saved mask paths (if save_dir given).
    """
    paths = sorted(glob(os.path.join(image_dir, "*.nii.gz")))
    if not paths:
        print(f"No .nii.gz files found in {image_dir}")
        return []

    saved = []
    os.makedirs(save_dir, exist_ok=True) if save_dir else None
    print(f"Found {len(paths)} files in {image_dir}")

    for i, p in enumerate(paths):
        base = os.path.basename(p).replace("case_", "pred_")
        out_path = os.path.join(save_dir, base) if save_dir else None
        img2d, pred = predict_one(model, p, save_path=out_path, resize=resize)
        if save_dir:
            saved.append(out_path)

        # visualize a few samples
        if i < max_visualize:
            visualize_triplet(img2d, pred, title=os.path.basename(p))

    return saved


# --------------------------
# Run examples
# --------------------------
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    model = load_model(CHECKPOINT_PATH, n_classes=NUM_CLASSES)

    # 1) Single-file prediction + visualization
    img2d, pred = predict_one(model, TEST_IMAGE, save_path=SAVE_SINGLE, resize=RESIZE)
    visualize_triplet(img2d, pred, title=os.path.basename(TEST_IMAGE))

    # 2) Batch predict an entire folder, save masks, and visualize a few
    _ = predict_folder(model, TEST_FOLDER, save_dir=SAVE_FOLDER, max_visualize=4, resize=RESIZE)