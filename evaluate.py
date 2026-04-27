"""
Top-1 and Top-5 accuracy evaluation over the full ImageNet-1K validation set
(50 000 images).

run_accuracy_eval() iterates the entire val DataLoader with no_grad, accumulates
correct top-1 and top-5 predictions, and saves results to
results/{label}_accuracy.json.

The returned dict includes 'top1_proportion' as a synonym for 'top1_accuracy'
so it can be passed directly to stats.compare_accuracy() without renaming.
"""

import json
import os
from typing import Any, Dict, Optional

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
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate top-1 and top-5 accuracy over the full validation set.

    Args:
        model:    Model in eval mode on *device*.
        loader:   DataLoader over the ImageNet validation split.
        device:   CUDA device string.
        label:    Identifier used for the output JSON filename.
        metadata: Optional reproducibility metadata dict (from meta.build_metadata()).
                  Stored under "metadata" key in the output JSON.

    Returns:
        dict with keys:
          label, correct, total, top1_accuracy, top1_proportion,
          correct_top5, top5_accuracy, top5_proportion,
          metadata (if provided)
    """
    model.eval()
    correct: int  = 0
    correct5: int = 0
    total: int    = 0
    n_batches     = len(loader)

    print(f"[{label}] Starting accuracy evaluation over {n_batches} batches…")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images  = images.to(device, dtype=torch.bfloat16, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)                              # (B, 1000)

            # Top-1
            preds    = outputs.argmax(dim=1)                    # (B,)
            correct += (preds == targets).sum().item()

            # Top-5: check whether true label is in top-5 predictions
            top5_preds  = outputs.topk(5, dim=1).indices        # (B, 5)
            correct5   += (top5_preds == targets.unsqueeze(1)).any(dim=1).sum().item()

            total += targets.size(0)

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_batches:
                running_acc  = correct  / total if total > 0 else 0.0
                running_acc5 = correct5 / total if total > 0 else 0.0
                print(
                    f"  Batch {batch_idx + 1:>4d}/{n_batches}  "
                    f"top-1: {running_acc:.4f}  top-5: {running_acc5:.4f}"
                    f"  ({correct}/{total})"
                )

    top1_acc: float = correct  / total if total > 0 else 0.0
    top5_acc: float = correct5 / total if total > 0 else 0.0

    print()
    print(f"{'=' * 55}")
    print(f"Accuracy results — {label}")
    print(f"{'=' * 55}")
    print(f"  Top-1 accuracy : {top1_acc:.4f}  ({correct}/{total})")
    print(f"  Top-5 accuracy : {top5_acc:.4f}  ({correct5}/{total})")
    print(f"{'=' * 55}")
    print()

    result: Dict[str, Any] = {
        # --- existing keys (unchanged) ---
        "label":           label,
        "correct":         correct,
        "total":           total,
        "top1_accuracy":   top1_acc,
        "top1_proportion": top1_acc,   # alias for stats.compare_accuracy()
        # --- new keys ---
        "correct_top5":    correct5,
        "top5_accuracy":   top5_acc,
        "top5_proportion": top5_acc,
    }

    if metadata is not None:
        result["metadata"] = metadata

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{label}_accuracy.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[{label}] Accuracy saved to {out_path}")

    return result
