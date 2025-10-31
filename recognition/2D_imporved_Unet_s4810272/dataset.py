"""
HipMRI 2D prostate segmentation dataset loader (simple version)

This loader does:
  Load paired .nii.gz image + mask files
  Ensure 2D format
  Z-score normalize MRI images
  Keep segmentation masks as integer class labels
  Resize both image + mask to a consistent size
  Return PyTorch tensors ready for a segmentation model
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# --------------------------
# Helper: Z-score normalize
# --------------------------
def _zscore(x, eps=1e-6):
    """
    Standardization: (x - mean) / std
    Used for MRI images to stabilize training.
    eps avoids division by zero when std=0.
    """
    mu = x.mean()
    sd = x.std()
    return (x - mu) / (sd + eps)

# ===================================================
# Dataset class
# ===================================================
class HipMRI(Dataset):
    """
    PyTorch Dataset for HipMRI 2D segmentation.

    Each __getitem__ returns:
      image: tensor shape (1, H, W), float32
      mask:  tensor shape (H, W),   int64 (class IDs)
    """

    def __init__(
        self,
        data_root,      # root folder path
        img_folder,
        seg_folder,
        resize=(256, 128),
        normalize=True,
        num_classes=6
    ):
        super().__init__()

        self.resize = resize
        self.normalize = normalize
        self.num_classes = num_classes

        self.img_dir = os.path.join(data_root, img_folder)
        self.seg_dir = os.path.join(data_root, seg_folder)

        # Find all image file paths
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.nii.gz")))
        if len(self.img_files) == 0:
            raise RuntimeError(f"No image files found in {self.img_dir}")

        # Make (image_path, mask_path) pairs
        self.pairs = []
        for img_path in self.img_files:
            name = os.path.basename(img_path)
            # Naming rule: case_XXX.nii.gz  → seg_XXX.nii.gz
            seg_path = os.path.join(self.seg_dir, name.replace("case_", "seg_"))

            if not os.path.exists(seg_path):
                raise FileNotFoundError(f"Mask missing for: {name}")

            self.pairs.append((img_path, seg_path))

        print(f"[HipMRI] Loaded {len(self.pairs)}")

    # required by torch.utils.data.DataLoader
    def __len__(self):
        return len(self.pairs)

    # -----------------------------------
    # Load one 2D .nii.gz file
    # -----------------------------------
    def _load_nifti(self, path, is_mask=False):
        img = nib.load(path).get_fdata(dtype=np.float32)

        if img.ndim != 2:
            raise ValueError(f"Expected 2D NIfTI but got shape {img.shape} at {path}")

        if is_mask:
            # Convert float → integer labels
            mask = np.rint(img).astype(np.int64)
            # Clip labels so they stay valid
            mask[(mask < 0) | (mask >= self.num_classes)] = 0
            return mask

        # Normalize MRI intensities if needed
        if self.normalize:
            img = _zscore(img)

        return img.astype(np.float32)

    # ---------------------------------
    # Return one training sample
    # ---------------------------------
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Load numpy arrays
        x = self._load_nifti(img_path, is_mask=False)  # (H, W)
        y = self._load_nifti(mask_path, is_mask=True)  # (H, W)

        # Convert to PyTorch tensors
        x = torch.from_numpy(x).unsqueeze(0)  # add channel → (1,H,W)
        y = torch.from_numpy(y).long()       # mask must be int64

        # Resize both image and mask if needed
        if self.resize is not None:
            H, W = self.resize
            if x.shape[1:] != (H, W):
                # image → bilinear (smooth)
                x = F.interpolate(
                    x.unsqueeze(0), size=(H, W),
                    mode="bilinear", align_corners=False
                ).squeeze(0)

                # mask → nearest (keeps class labels)
                y = F.interpolate(
                    y.unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W), mode="nearest"
                ).squeeze(0).squeeze(0).long()

        return x.float(), y


# ===================================================
# DataLoader helper 
# ===================================================
def make_loaders(
    data_root,
    batch_size=8,
    num_workers=4,
    resize=(256, 128),
    normalize=True,
    num_classes=6
):
    """
    Build train, validation, and test DataLoaders using explicit folder names.
    """

    train_ds = HipMRI(
        data_root,
        img_folder="keras_slices_train",
        seg_folder="keras_slices_seg_train",
        resize=resize,
        normalize=normalize,
        num_classes=num_classes
    )

    val_ds = HipMRI(
        data_root,
        img_folder="keras_slices_validate",
        seg_folder="keras_slices_seg_validate",
        resize=resize,
        normalize=normalize,
        num_classes=num_classes
    )

    test_ds = HipMRI(
        data_root,
        img_folder="keras_slices_test",
        seg_folder="keras_slices_seg_test",
        resize=resize,
        normalize=normalize,
        num_classes=num_classes
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader