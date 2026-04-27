"""
Reproducibility metadata helpers for ToMe-Flash experiment scripts.

build_metadata() produces a JSON-serialisable dict that is embedded under
the "metadata" key in every benchmark/accuracy/phase-stats output file.
"""

import datetime
import os
import subprocess
import sys
from typing import Any, Dict, Optional

import torch


def _git_commit() -> str:
    """Return the short HEAD commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_metadata(
    script_name: str,
    args_dict: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    n_warmup: Optional[int] = None,
    n_trials: Optional[int] = None,
    seed: Optional[int] = None,
    deterministic: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build a reproducibility metadata block for output JSONs.

    Args:
        script_name:   Basename of the calling script (e.g. 'run_phase1.py').
        args_dict:     vars(args) from argparse, or None.
        device:        CUDA device string used for the run.
        batch_size:    Batch size used, if known.
        num_workers:   DataLoader worker count, if known.
        n_warmup:      Warm-up iterations, if relevant.
        n_trials:      Timed-trial count, if relevant.
        seed:          Random seed, if set.
        deterministic: Whether torch backends were set to deterministic mode.

    Returns:
        JSON-serialisable dict with reproducibility fields.
    """
    cuda_version: str = torch.version.cuda or "N/A"
    device_name: str = "cpu"
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(idx)

    meta: Dict[str, Any] = {
        "timestamp_utc":   datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit":      _git_commit(),
        "python_version":  sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version":    cuda_version,
        "device_name":     device_name,
        "script_name":     script_name,
    }
    if batch_size is not None:
        meta["batch_size"] = batch_size
    if num_workers is not None:
        meta["num_workers"] = num_workers
    if n_warmup is not None:
        meta["n_warmup"] = n_warmup
    if n_trials is not None:
        meta["n_trials"] = n_trials
    if seed is not None:
        meta["seed"] = seed
    if deterministic is not None:
        meta["deterministic"] = deterministic
    if args_dict is not None:
        meta["args"] = args_dict

    return meta
