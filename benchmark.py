"""
Timed inference benchmark for the ToMe-Flash pipeline.

run_benchmark() performs n_warmup un-timed forward passes to warm up the GPU
and CUDA caches, then runs n_trials timed trials collecting per-trial throughput
(images/sec) and peak GPU memory (GB).

CRITICAL timing note:
  torch.cuda.synchronize() must be called BOTH before recording t0 AND after the
  forward pass. Without both calls, CUDA's asynchronous execution model means the
  kernel may not have started (or finished) at the points we intend to measure.
"""

import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import trim_mean as _scipy_trim_mean
from torch.utils.data import DataLoader

PROJECT_ROOT: str = "/scratch/rsp9219/project"
RESULTS_DIR: str  = os.path.join(PROJECT_ROOT, "results")


def _robust_stats(values: List[float], prefix: str) -> Dict[str, Any]:
    """Compute robust descriptive statistics for a list of floats.

    Returns a dict of new keys prefixed with *prefix* (e.g. 'throughput'):
      median_{prefix}, iqr_{prefix}, p5_{prefix}, p95_{prefix},
      trimmed_mean_{prefix}, num_outliers_iqr_rule_{prefix}
    """
    arr = np.array(values, dtype=float)
    q1, q3  = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr     = q3 - q1
    p5      = float(np.percentile(arr, 5))
    p95     = float(np.percentile(arr, 95))
    median  = float(np.median(arr))
    # 10% trimmed mean (drops bottom and top 10%)
    trimmed = float(_scipy_trim_mean(arr, 0.10))
    # IQR rule: outlier if < Q1 - 1.5*IQR or > Q3 + 1.5*IQR
    n_out   = int(np.sum((arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)))

    return {
        f"median_{prefix}":                  median,
        f"iqr_{prefix}":                     float(iqr),
        f"p5_{prefix}":                      p5,
        f"p95_{prefix}":                     p95,
        f"trimmed_mean_{prefix}":            trimmed,
        f"num_outliers_iqr_rule_{prefix}":   n_out,
    }


def run_benchmark(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    n_warmup: int = 30,
    n_trials: int = 30,
    label: str = "baseline",
    metadata: Optional[Dict[str, Any]] = None,
    profile_dir: Optional[str] = None,
    profile_wait: int = 1,
    profile_warmup: int = 1,
    profile_active: int = 6,
    profile_repeat: int = 1,
    profile_record_shapes: bool = True,
    profile_memory: bool = True,
    profile_with_stack: bool = False,
) -> Dict[str, Any]:
    """Run a timed inference benchmark and return per-trial statistics.

    Args:
        model:    Model in eval mode on *device*.
        loader:   DataLoader supplying (images, labels) batches.
        device:   CUDA device string (e.g. 'cuda' or 'cuda:0').
        n_warmup: Number of un-timed warm-up forward passes.
        n_trials: Number of timed trials; each trial = one forward pass over
                  one batch from *loader*.
        label:    Identifier used for the output JSON filename and printed table.
        metadata: Optional reproducibility metadata dict (from meta.build_metadata()).
                  Stored under "metadata" key in the output JSON.
        profile_dir: Optional output directory for PyTorch Profiler traces.
                     If None, profiling is disabled.
        profile_wait/profile_warmup/profile_active/profile_repeat:
                     torch.profiler.schedule(...) controls.
        profile_record_shapes/profile_memory/profile_with_stack:
                     Additional profiler detail controls.

    Returns:
        dict with keys (existing keys preserved for backward compatibility):
          label, throughputs, peak_memories,
          mean_throughput, std_throughput, mean_memory, std_memory,
          batch_size, n_trials,
          elapsed_ms (per-trial list),
          median_throughput, iqr_throughput, p5_throughput, p95_throughput,
          trimmed_mean_throughput, num_outliers_iqr_rule_throughput,
          median_memory, iqr_memory, p5_memory, p95_memory,
          trimmed_mean_memory, num_outliers_iqr_rule_memory,
          metadata (if provided)
    """
    model.eval()
    loader_iter = iter(loader)
    batch_size: int = loader.batch_size or 64  # type: ignore[arg-type]

    # ------------------------------------------------------------------ warmup
    print(f"[{label}] Warming up ({n_warmup} batches)…")
    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                images, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                images, _ = next(loader_iter)
            images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
            _ = model(images)
    torch.cuda.synchronize(device)

    # ------------------------------------------------------------------ timed trials
    throughputs:   List[float] = []
    peak_memories: List[float] = []
    elapsed_ms:    List[float] = []
    traces_written = False

    prof = None
    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        total_profile_steps = profile_repeat * (profile_wait + profile_warmup + profile_active)
        if n_trials < total_profile_steps:
            print(
                f"[{label}] WARNING: n_trials={n_trials} is smaller than profiler schedule "
                f"steps={total_profile_steps}. Increase n_trials or reduce schedule to ensure "
                "an active trace window is captured."
            )

        print(
            f"[{label}] Profiling enabled. Writing traces to: {profile_dir} "
            f"(wait={profile_wait}, warmup={profile_warmup}, active={profile_active}, repeat={profile_repeat})"
        )
        prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=profile_wait,
                warmup=profile_warmup,
                active=profile_active,
                repeat=profile_repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir, worker_name=label),
            record_shapes=profile_record_shapes,
            profile_memory=profile_memory,
            with_stack=profile_with_stack,
        )
        prof.start()

    print(f"[{label}] Running {n_trials} timed trials…")
    with torch.no_grad():
        for _ in range(n_trials):
            try:
                images, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                images, _ = next(loader_iter)

            images       = images.to(device, dtype=torch.bfloat16, non_blocking=True)
            actual_batch = images.size(0)

            torch.cuda.reset_peak_memory_stats(device)

            # Synchronize before starting the clock
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            _ = model(images)

            # Synchronize after the forward pass so elapsed reflects GPU completion
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0

            if prof is not None:
                prof.step()
                traces_written = True

            peak_mem_gb: float = torch.cuda.max_memory_allocated(device) / 1e9
            throughput:  float = actual_batch / elapsed

            throughputs.append(throughput)
            peak_memories.append(peak_mem_gb)
            elapsed_ms.append(elapsed * 1000.0)

    if prof is not None:
        prof.stop()
        if traces_written:
            print(f"[{label}] Profiler trace export complete: {profile_dir}")
        else:
            print(f"[{label}] Profiler ran but no trace steps were recorded.")

    # ------------------------------------------------------------------ statistics
    mean_tp  = statistics.mean(throughputs)
    std_tp   = statistics.stdev(throughputs)
    mean_mem = statistics.mean(peak_memories)
    std_mem  = statistics.stdev(peak_memories)

    robust_tp  = _robust_stats(throughputs,   "throughput")
    robust_mem = _robust_stats(peak_memories, "memory")

    # ------------------------------------------------------------------ print summary
    print()
    print(f"{'=' * 60}")
    print(f"Benchmark results — {label}")
    print(f"{'=' * 60}")
    print(f"  Throughput : {mean_tp:>9.1f} ± {std_tp:.1f}  images/sec")
    print(f"  Median tp  : {robust_tp['median_throughput']:>9.1f}  (IQR {robust_tp['iqr_throughput']:.1f})")
    print(f"  P5 / P95   : {robust_tp['p5_throughput']:.1f} / {robust_tp['p95_throughput']:.1f}")
    print(f"  Peak VRAM  : {mean_mem:>9.3f} ± {std_mem:.3f}  GB")
    print(f"  Batch size : {batch_size}")
    print(f"  Trials     : {n_trials}  ({robust_tp['num_outliers_iqr_rule_throughput']} throughput outliers by IQR rule)")
    print(f"{'=' * 60}")
    print()

    result: Dict[str, Any] = {
        # --- existing keys (unchanged) ---
        "label":           label,
        "throughputs":     throughputs,
        "peak_memories":   peak_memories,
        "mean_throughput": mean_tp,
        "std_throughput":  std_tp,
        "mean_memory":     mean_mem,
        "std_memory":      std_mem,
        "batch_size":      batch_size,
        "n_trials":        n_trials,
        # --- new keys ---
        "elapsed_ms":      elapsed_ms,
        **robust_tp,
        **robust_mem,
    }

    if metadata is not None:
        result["metadata"] = metadata

    # ------------------------------------------------------------------ save JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{label}_benchmark.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[{label}] Benchmark saved to {out_path}")

    return result
