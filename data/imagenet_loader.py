"""
ImageNet-1K validation DataLoader for the ToMe-Flash benchmark pipeline.

Provides get_imagenet_val_loader() with DeiT-B/16-compatible preprocessing
(Resize(256) → CenterCrop(224) → ToTensor → ImageNet normalize) and a
get_fast_loader() variant that stops after max_batches for quick smoke tests.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Path constants — only hardcoded paths live here
# ---------------------------------------------------------------------------
IMAGENET_VAL_PATH: str = "/imagenet/val"

# DeiT-B/16 training preprocessing (must match exactly)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def get_imagenet_val_loader(
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = True,
    data_path: str = IMAGENET_VAL_PATH,
) -> DataLoader:
    """Return a DataLoader over the full ImageNet-1K validation set (50 000 images).

    Args:
        batch_size:  Number of images per batch.
        num_workers: Number of parallel data-loading workers.
        pin_memory:  Whether to use pinned (page-locked) CPU memory for faster
                     host-to-device transfers.
        data_path:   Root directory of the ImageNet val split
                     (expects ILSVRC2012_img_val/ subfolder layout with class dirs).

    Returns:
        A DataLoader that yields (images, labels) tensors.
    """
    dataset = datasets.ImageFolder(root=data_path, transform=_VAL_TRANSFORMS)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader


def get_fast_loader(
    max_batches: int = 10,
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = True,
    data_path: str = IMAGENET_VAL_PATH,
) -> DataLoader:
    """Return a DataLoader limited to the first max_batches batches for smoke tests.

    Internally takes a Subset of the first (max_batches * batch_size) samples so
    iteration terminates quickly without changing any other loader behaviour.

    Args:
        max_batches: Maximum number of batches to expose.
        batch_size:  Number of images per batch.
        num_workers: Number of parallel data-loading workers.
        pin_memory:  Whether to use pinned CPU memory.
        data_path:   Root directory of the ImageNet val split.

    Returns:
        A DataLoader that yields at most max_batches batches.
    """
    dataset = datasets.ImageFolder(root=data_path, transform=_VAL_TRANSFORMS)
    n_samples = min(max_batches * batch_size, len(dataset))
    subset = Subset(dataset, list(range(n_samples)))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader
