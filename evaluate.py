"""
Top-1 accuracy evaluation over the full ImageNet-1K validation set (50 000 images).

run_accuracy_eval() iterates the entire val DataLoader with no_grad, accumulates
correct top-1 predictions, and saves results to results/{label}_accuracy.json.
The returned dict includes 'top1_proportion' as a synonym for 'top1_accuracy'
so it can be passed directly to stats.compare_accuracy() without renaming.
"""

import json
import os
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT: str = "/scratch/rsp9219/project"
RESULTS_DIR: str  = os.path.join(PROJECT_ROOT, "results")


def run_accuracy_eval(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    label: str = "baseline",
) -> Dict[str, Any]:
    """Evaluate top-1 accuracy over the full validation set.

    Args:
        model:  Model in eval mode on *device*.
        loader: DataLoader over the ImageNet validation split.
        device: CUDA device string.
        label:  Identifier used for the output JSON filename.

    Returns:
        dict with keys:
          label, correct, total, top1_accuracy, top1_proportion
    """
    model.eval()
    correct: int = 0
    total: int   = 0
    n_batches    = len(loader)

    print(f"[{label}] Starting accuracy evaluation over {n_batches} batches…")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images  = images.to(device, dtype=torch.bfloat16, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs  = model(images)                    # (B, 1000)
            preds    = outputs.argmax(dim=1)            # (B,)
            correct += (preds == targets).sum().item()
            total   += targets.size(0)

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_batches:
                running_acc = correct / total if total > 0 else 0.0
                print(
                    f"  Batch {batch_idx + 1:>4d}/{n_batches}  "
                    f"running top-1: {running_acc:.4f} ({correct}/{total})"
                )

    top1_acc: float = correct / total if total > 0 else 0.0

    print()
    print(f"{'=' * 50}")
    print(f"Accuracy results — {label}")
    print(f"{'=' * 50}")
    print(f"  Top-1 accuracy : {top1_acc:.4f}  ({correct}/{total})")
    print(f"{'=' * 50}")
    print()

    result: Dict[str, Any] = {
        "label":           label,
        "correct":         correct,
        "total":           total,
        "top1_accuracy":   top1_acc,
        "top1_proportion": top1_acc,   # alias for stats.compare_accuracy()
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{label}_accuracy.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[{label}] Accuracy saved to {out_path}")

    return result
