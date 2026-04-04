"""
Phase 1 entry point: baseline DeiT-B/16 inference pipeline on ImageNet-1K.

Runs the full benchmark (throughput + peak VRAM, 30 warm-up + 30 timed trials)
and accuracy evaluation (top-1 over 50 000 val images) for the unmodified baseline
model, saving results to results/baseline_benchmark.json and
results/baseline_accuracy.json.

Usage:
    python run_phase1.py [--batch_size 64] [--num_workers 8] [--fast] [--device cuda]

Flags:
    --fast     Smoke-test mode: only 10 batches for both benchmark and accuracy eval.
    --device   Override compute device (default: cuda).
"""

import argparse
import os
import sys
import warnings

import torch

PROJECT_ROOT: str = "/scratch/rsp9219/project"
RESULTS_DIR: str  = os.path.join(PROJECT_ROOT, "results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: baseline DeiT-B/16 benchmark on ImageNet-1K val"
    )
    parser.add_argument("--batch_size",  type=int,  default=64,     help="Batch size (default: 64)")
    parser.add_argument("--num_workers", type=int,  default=8,      help="DataLoader workers (default: 8)")
    parser.add_argument("--fast",        action="store_true",       help="Smoke-test with only 10 batches")
    parser.add_argument("--device",      type=str,  default="cuda", help="Compute device (default: cuda)")
    return parser.parse_args()


def print_env_info(device: str) -> None:
    """Print PyTorch, CUDA, and GPU information."""
    print("=" * 60)
    print("Environment")
    print("=" * 60)
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA        : {torch.version.cuda}")
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"  GPU         : {torch.cuda.get_device_name(idx)}")
        total_mem = torch.cuda.get_device_properties(idx).total_memory / 1e9
        print(f"  VRAM total  : {total_mem:.1f} GB")
    else:
        print("  GPU         : (not available — running on CPU)")
    print("=" * 60)
    print()


def main() -> None:
    args = parse_args()

    # Ensure results directory exists before anything writes to it
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Environment info
    # ------------------------------------------------------------------
    print_env_info(args.device)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    from models.baseline import load_baseline_model, get_model_info

    print("Loading baseline model (DeiT-B/16, pretrained)…")
    model = load_baseline_model(device=args.device)
    get_model_info(model)

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    from data.imagenet_loader import get_imagenet_val_loader, get_fast_loader

    if args.fast:
        print(f"[FAST MODE] Using only 10 batches (batch_size={args.batch_size}).")
        loader = get_fast_loader(
            max_batches=10,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        n_warmup  = 3
        n_trials  = 10
    else:
        loader = get_imagenet_val_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        n_warmup  = 30
        n_trials  = 30

    print(f"DataLoader: {len(loader)} batches, batch_size={args.batch_size}, "
          f"num_workers={args.num_workers}\n")

    # ------------------------------------------------------------------
    # Benchmark (throughput + memory)
    # ------------------------------------------------------------------
    from benchmark import run_benchmark

    bench_results = run_benchmark(
        model=model,
        loader=loader,
        device=args.device,
        n_warmup=n_warmup,
        n_trials=n_trials,
        label="baseline",
    )

    # ------------------------------------------------------------------
    # Accuracy evaluation
    # ------------------------------------------------------------------
    from evaluate import run_accuracy_eval

    # Reload a fresh full-length loader for accuracy eval (even in fast mode,
    # reuse the same limited loader so the smoke test finishes quickly)
    if args.fast:
        acc_loader = get_fast_loader(
            max_batches=10,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        acc_loader = get_imagenet_val_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    acc_results = run_accuracy_eval(
        model=model,
        loader=acc_loader,
        device=args.device,
        label="baseline",
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    top1 = acc_results["top1_accuracy"]

    print("=" * 60)
    print("Phase 1 — Final Summary (baseline DeiT-B/16)")
    print("=" * 60)
    print(f"  Throughput : {bench_results['mean_throughput']:.1f} ± "
          f"{bench_results['std_throughput']:.1f}  images/sec")
    print(f"  Peak VRAM  : {bench_results['mean_memory']:.3f} ± "
          f"{bench_results['std_memory']:.3f}  GB")
    print(f"  Top-1 acc  : {top1:.4f}  ({acc_results['correct']}/{acc_results['total']})")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Accuracy sanity check — DeiT-B/16 expected ~81.8%
    # Only meaningful on the full val set; skip in fast mode
    # ------------------------------------------------------------------
    if not args.fast:
        if not (0.79 <= top1 <= 0.82):
            warnings.warn(
                f"WARNING: Top-1 accuracy {top1:.4f} is outside the expected range "
                f"[0.79, 0.82] for DeiT-B/16. "
                "Check preprocessing transforms and model checkpoint.",
                stacklevel=2,
            )
        else:
            print(f"Accuracy check PASSED: {top1:.4f} is within [0.79, 0.82].")
    else:
        print("[FAST MODE] Skipping accuracy range check (subset only).")

    print("\nPhase 1 complete.")
    print(f"Results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
