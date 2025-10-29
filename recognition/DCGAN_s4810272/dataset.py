# dataset.py
import os
import glob
from typing import Optional, Tuple, Callable

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd + eps)


def _as_2d(arr: np.ndarray) -> np.ndarray:
    """
    Ensures a 2D array. For NIfTI saved as a single-slice 3D (H,W,1),
    squeeze the singleton. If truly 3D, take the central slice.
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[..., 0]
        # fallback: central slice along last axis
        mid = arr.shape[-1] // 2
        return arr[..., mid]
    raise ValueError(f"Expected 2D/3D array, got shape {arr.shape}")


class HipMRIDataset(Dataset):
    """
    HipMRI 2D prostate segmentation dataset.
    Loads paired .nii.gz image/mask slices for a given split.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        resize: Optional[Tuple[int, int]] = (256, 128),
        normalize: bool = True,
        transform: Optional[Callable] = None,
        num_classes: int = 6,
        img_glob: str = "*.nii.gz",
        img_folder_map=None,
        seg_folder_map=None,
    ):
        """
        Args:
            data_root: path that contains keras_slices_* folders
            split: 'train' | 'val' | 'test'
            resize: output (H, W) or None to keep original
            normalize: z-score images
            transform: optional callable(img_t, mask_t) -> (img_t, mask_t)
            num_classes: labels outside [0..num_classes-1] are clipped to 0
            img_glob: pattern for image files
            img_folder_map: optional dict to override image split folder names
            seg_folder_map: optional dict to override mask split folder names
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.resize = resize
        self.normalize = normalize
        self.transform = transform
        self.num_classes = num_classes

        img_map = img_folder_map or {
            "train": "keras_slices_train",
            "val":   "keras_slices_validate",
            "test":  "keras_slices_test",
        }
        seg_map = seg_folder_map or {
            "train": "keras_slices_seg_train",
            "val":   "keras_slices_seg_validate",
            "test":  "keras_slices_seg_test",
        }

        self.img_dir = os.path.join(data_root, img_map[split])
        self.seg_dir = os.path.join(data_root, seg_map[split])

        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, img_glob)))
        if not self.img_files:
            raise RuntimeError(f"No images found in {self.img_dir} with pattern {img_glob}")

        # Pair masks by replacing the expected prefix; adjust here if your naming differs
        self.pairs = []
        for ip in self.img_files:
            name = os.path.basename(ip)
            # example: case_XXX.nii.gz -> seg_XXX.nii.gz
            seg_name = name.replace("case_", "seg_")
            mp = os.path.join(self.seg_dir, seg_name)
            if not os.path.exists(mp):
                raise FileNotFoundError(f"Missing mask for {name}: {mp}")
            self.pairs.append((ip, mp))

        print(f"[HipMRISlices] {split}: {len(self.pairs)} files")

    def __len__(self):
        return len(self.pairs)

    def _load_nifti_2d(self, path: str, is_mask: bool) -> np.ndarray:
        arr = nib.load(path).get_fdata(dtype=np.float32)
        arr2d = _as_2d(arr)
        if is_mask:
            # round to nearest integer class indices
            m = np.rint(arr2d).astype(np.int64)
            # clip any out-of-range labels back to background
            m[(m < 0) | (m >= self.num_classes)] = 0
            return m
        # image
        x = arr2d.astype(np.float32)
        if self.normalize:
            x = _zscore(x)
        return x

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        x = self._load_nifti_2d(img_path, is_mask=False)  # (H,W)
        y = self._load_nifti_2d(mask_path, is_mask=True)  # (H,W)

        # to torch, add channel to image
        x = torch.from_numpy(x).unsqueeze(0)  # (1,H,W)
        y = torch.from_numpy(y).long()        # (H,W)

        # optional resize
        if self.resize is not None:
            H, W = self.resize
            if x.shape[1:] != (H, W):
                x = F.interpolate(x.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
                y = F.interpolate(y.unsqueeze(0).unsqueeze(0).float(), size=(H, W), mode="nearest").squeeze(0).squeeze(0).long()

        # user-supplied transforms (ensure they return torch tensors)
        if self.transform is not None:
            x, y = self.transform(x, y)

        return x.float(), y


def make_loaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    resize: Optional[Tuple[int, int]] = (256, 128),
    normalize: bool = True,
    num_classes: int = 6,
    transform_train: Optional[Callable] = None,
    transform_eval: Optional[Callable] = None,
):
    """
    Builds train/val/test DataLoaders for the HipMRI 2D splits.
    """
    train_ds = HipMRIDataset(
        data_root, split="train", resize=resize, normalize=normalize,
        transform=transform_train, num_classes=num_classes
    )
    val_ds = HipMRIDataset(
        data_root, split="val", resize=resize, normalize=normalize,
        transform=transform_eval, num_classes=num_classes
    )
    test_ds = HipMRIDataset(
        data_root, split="test", resize=resize, normalize=normalize,
        transform=transform_eval, num_classes=num_classes
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader