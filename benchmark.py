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
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT: str = "/scratch/rsp9219/project"
RESULTS_DIR: str  = os.path.join(PROJECT_ROOT, "results")


def run_benchmark(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    n_warmup: int = 30,
    n_trials: int = 30,
    label: str = "baseline",
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

    Returns:
        dict with keys:
          label, throughputs, peak_memories, mean_throughput, std_throughput,
          mean_memory, std_memory, batch_size, n_trials
    """
    model.eval()
    loader_iter = iter(loader)
    batch_size: int = loader.batch_size or 64  # type: ignore[arg-type]

    # ------------------------------------------------------------------ warmup
    print(f"[{label}] Warming up ({n_warmup} batches)…")
    with torch.no_grad():
        for i in range(n_warmup):
            try:
                images, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                images, _ = next(loader_iter)
            images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
            _ = model(images)
    torch.cuda.synchronize(device)

    # ------------------------------------------------------------------ timed trials
    throughputs: List[float] = []
    peak_memories: List[float] = []

    print(f"[{label}] Running {n_trials} timed trials…")
    with torch.no_grad():
        for trial in range(n_trials):
            try:
                images, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                images, _ = next(loader_iter)

            images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
            actual_batch = images.size(0)

            # Reset peak memory counter before this trial
            torch.cuda.reset_peak_memory_stats(device)

            # Synchronize before starting the clock
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            _ = model(images)

            # Synchronize after the forward pass so elapsed reflects GPU completion
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0

            peak_mem_gb: float = torch.cuda.max_memory_allocated(device) / 1e9
            throughput: float  = actual_batch / elapsed

            throughputs.append(throughput)
            peak_memories.append(peak_mem_gb)

    # ------------------------------------------------------------------ statistics
    import statistics
    mean_tp  = statistics.mean(throughputs)
    std_tp   = statistics.stdev(throughputs)
    mean_mem = statistics.mean(peak_memories)
    std_mem  = statistics.stdev(peak_memories)

    # ------------------------------------------------------------------ print summary
    print()
    print(f"{'=' * 60}")
    print(f"Benchmark results — {label}")
    print(f"{'=' * 60}")
    print(f"  Throughput : {mean_tp:>9.1f} ± {std_tp:.1f}  images/sec")
    print(f"  Peak VRAM  : {mean_mem:>9.3f} ± {std_mem:.3f}  GB")
    print(f"  Batch size : {batch_size}")
    print(f"  Trials     : {n_trials}")
    print(f"{'=' * 60}")
    print()

    result: Dict[str, Any] = {
        "label":           label,
        "throughputs":     throughputs,
        "peak_memories":   peak_memories,
        "mean_throughput": mean_tp,
        "std_throughput":  std_tp,
        "mean_memory":     mean_mem,
        "std_memory":      std_mem,
        "batch_size":      batch_size,
        "n_trials":        n_trials,
    }

    # ------------------------------------------------------------------ save JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{label}_benchmark.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[{label}] Benchmark saved to {out_path}")

    return result
