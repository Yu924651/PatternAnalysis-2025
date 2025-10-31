"""
  Predict + Visualize for HipMRI
  loads a trained Improved2DUNet model,
  performs segmentation prediction on a single MRI slice (.nii.gz file)
  visualizes both the predicted mask and the input image.
"""
import os
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from dataset import _zscore # remove if running in Colab and model is defined elsewhere
from modules import Improved2DUNet  # remove if running in Colab and model is defined elsewhere


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
RESIZE = (256, 128)  # must match training
# path
CHECKPOINT_PATH = "/content/drive/My Drive/checkpoints/best_model.pth"
TEST_IMAGE = "/content/drive/My Drive/keras_slices_data/keras_slices_test/case_040_week_0_slice_0.nii.gz"
GROUND_TRUTH = "/content/drive/My Drive/keras_slices_data/keras_slices_seg_test/seg_040_week_0_slice_0.nii.gz"


def load_model(checkpoint_path: str, n_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """
    Loads the trained Improved2DUNet model from saved file.
    Moves it to the device and sets it to evaluation mode.
    """
    model = Improved2DUNet(in_channels=1, n_classes=n_classes, base=32, p_drop=0.2)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model

@torch.no_grad()
def predict_one(model: torch.nn.Module, image_path: str, resize: tuple = RESIZE):
    """
    Predicts the segmentation mask for a single 2D .nii.gz image (no saving).
    """
    nii = nib.load(image_path)
    img = nii.get_fdata(dtype=np.float32)

    # Ensure image is 2D
    if img.ndim != 2:
        raise ValueError(f"Expected 2D NIfTI but got shape {img.shape} at {image_path}")

    # Normalize
    img_norm = _zscore(img)
    img_t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,W)

    # Resize to training if needed
    if resize is not None and img_t.shape[2:] != resize:
        img_t = F.interpolate(img_t, size=resize, mode="bilinear", align_corners=False)

    # prediction
    logits = model(img_t)
    pred = torch.argmax(logits, dim=1)  # (1,H,W)

    # Back to numpy for visualization
    pred_mask = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    return img_norm, pred_mask


def load_ground_truth(mask_path: str, resize: tuple = RESIZE):
    """
    Load a 2D NIfTI ground-truth mask, cast to integer labels, and (optionally) resize.
    """
    nii = nib.load(mask_path)
    mask = nii.get_fdata(dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D NIfTI mask but got shape {mask.shape} at {mask_path}")

    mask = np.rint(mask).astype(np.int64)

    # Resize via nearest neighbor to preserve class IDs
    if resize is not None and mask.shape != resize:
        mask_t = torch.from_numpy(mask[None, None].astype(np.float32))
        mask_t = F.interpolate(mask_t, size=resize, mode="nearest")
        mask = mask_t.squeeze().numpy().astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    return mask

def visualize_prediction(img2d: np.ndarray, mask2d: np.ndarray,  gt_mask: np.ndarray, title: str = ""):
    """
    Displays three panels side-by-side:
      1. Original input MRI slice
      2. Predicted segmentation mask
      3. Ground-truth mask
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img2d, cmap="gray")
    plt.title("Input MRI")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask2d, cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES - 1)
    plt.title("Predicted Mask")
    plt.axis("off")

    # Ground truth mask
    plt.subplot(1, 3, 3)
    plt.imshow(gt_mask, cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES - 1)
    plt.title("Ground Truth Mask")
    plt.axis("off")
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# Run single prediction
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    model = load_model(CHECKPOINT_PATH, n_classes=NUM_CLASSES)

    img2d, pred = predict_one(model, TEST_IMAGE, resize=RESIZE)
    # Load ground truth for comparison
    gt_mask = load_ground_truth(GROUND_TRUTH, resize=RESIZE)
    visualize_prediction(img2d, pred, gt_mask, title=os.path.basename(TEST_IMAGE))