"""
run_flash_attention.py

FlashAttention-only entry point: FlashAttention-2-only DeiT-B/16 benchmark on ImageNet-1K.

Runs the full benchmark (throughput + peak VRAM) and accuracy evaluation
(top-1 and top-5 over 50 000 val images) for the FA-only condition, then performs
statistical comparisons against the Phase 1 baseline results.

Results saved to:
  results/flash_only_benchmark.json
  results/flash_only_accuracy.json
  results/phase3_stats.json

Usage:
    python run_flash_attention.py [--batch_size 256] [--num_workers 8] [--fast] [--device cuda]

Flags:
    --fast     Smoke-test mode: only 10 batches for benchmark and accuracy eval.
    --device   Override compute device (default: cuda).
"""

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT: str = "/scratch/rsp9219/project"
RESULTS_DIR: str  = os.path.join(PROJECT_ROOT, "results")

LABEL = "flash_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlashAttention-only: FlashAttention-2-only DeiT-B/16 benchmark on ImageNet-1K val"
    )
    parser.add_argument("--batch_size",  type=int,  default=256,   help="Batch size (default: 256)")
    parser.add_argument("--num_workers", type=int,  default=8,     help="DataLoader workers (default: 8)")
    parser.add_argument("--fast",        action="store_true",      help="Smoke-test with only 10 batches")
    parser.add_argument("--device",      type=str,  default="cuda", help="Compute device (default: cuda)")
    parser.add_argument("--n_warmup",    type=int,  default=100,   help="Warm-up batches (default: 100)")
    parser.add_argument("--n_trials",    type=int,  default=100,   help="Timed trial batches (default: 100)")
    return parser.parse_args()


def print_env_info(device: str) -> None:
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


def load_baseline_results(results_dir: str) -> tuple:
    """Load Phase 1 baseline JSON results.

    Returns:
        (baseline_bench dict, baseline_acc dict)

    Raises:
        FileNotFoundError: If baseline JSON files are missing — run run_phase1.py first.
    """
    bench_path = os.path.join(results_dir, "baseline_benchmark.json")
    acc_path   = os.path.join(results_dir, "baseline_accuracy.json")

    for p in (bench_path, acc_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Baseline result not found: {p}\n"
                "Run run_phase1.py first to generate baseline results."
            )

    with open(bench_path) as fh:
        baseline_bench = json.load(fh)
    with open(acc_path) as fh:
        baseline_acc = json.load(fh)

    print("Loaded Phase 1 baseline results:")
    print(f"  Throughput : {baseline_bench['mean_throughput']:.1f} ± "
          f"{baseline_bench['std_throughput']:.1f} img/sec")
    print(f"  Memory     : {baseline_bench['mean_memory']:.3f} ± "
          f"{baseline_bench['std_memory']:.3f} GB")
    print(f"  Top-1 acc  : {baseline_acc['top1_accuracy']:.4f} "
          f"({baseline_acc['correct']}/{baseline_acc['total']})")
    print()
    return baseline_bench, baseline_acc


def main() -> None:
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print_env_info(args.device)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    from models.flash_only import load_flash_model

    print("Loading FA-only model (DeiT-B/16 + FlashAttention-2)…")
    model = load_flash_model(device=args.device)
    print()

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    from data.imagenet_loader import get_imagenet_val_loader, get_fast_loader

    if args.fast:
        print(f"[FAST MODE] Using only 10 batches (batch_size={args.batch_size}).")
        n_warmup     = 3
        n_trials     = 10
        bench_loader = get_fast_loader(
            max_batches=10, batch_size=args.batch_size, num_workers=args.num_workers,
        )
        acc_loader   = get_fast_loader(
            max_batches=10, batch_size=args.batch_size, num_workers=args.num_workers,
        )
    else:
        n_warmup     = args.n_warmup
        n_trials     = args.n_trials
        bench_loader = get_imagenet_val_loader(
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        acc_loader   = get_imagenet_val_loader(
            batch_size=args.batch_size, num_workers=args.num_workers,
        )

    print(f"DataLoader: {len(bench_loader)} batches, batch_size={args.batch_size}, "
          f"num_workers={args.num_workers}\n")

    # ------------------------------------------------------------------
    # Reproducibility metadata
    # ------------------------------------------------------------------
    from meta import build_metadata

    meta = build_metadata(
        script_name="run_flash_attention.py",
        args_dict=vars(args),
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_warmup=n_warmup,
        n_trials=n_trials,
    )

    # ------------------------------------------------------------------
    # Benchmark (throughput + memory)
    # ------------------------------------------------------------------
    from benchmark import run_benchmark

    flash_bench = run_benchmark(
        model=model,
        loader=bench_loader,
        device=args.device,
        n_warmup=n_warmup,
        n_trials=n_trials,
        label=LABEL,
        metadata=meta,
    )

    # ------------------------------------------------------------------
    # Accuracy evaluation
    # ------------------------------------------------------------------
    from evaluate import run_accuracy_eval

    flash_acc = run_accuracy_eval(
        model=model,
        loader=acc_loader,
        device=args.device,
        label=LABEL,
        metadata=meta,
    )

    # ------------------------------------------------------------------
    # Statistical comparisons vs Phase 1 baseline
    # ------------------------------------------------------------------
    baseline_bench, baseline_acc = load_baseline_results(RESULTS_DIR)

    from stats import compare_throughput, compare_memory, compare_accuracy, print_stats_table

    tp_stats  = compare_throughput(baseline_bench, flash_bench)
    mem_stats = compare_memory(baseline_bench, flash_bench)
    acc_stats = compare_accuracy(baseline_acc, flash_acc)

    print_stats_table(baseline_bench, baseline_acc, flash_bench, flash_acc, LABEL)

    # ------------------------------------------------------------------
    # Phase 3 summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("FlashAttention — Final Summary (FA-only DeiT-B/16)")
    print("=" * 60)
    print(f"  Throughput : {flash_bench['mean_throughput']:.1f} ± "
          f"{flash_bench['std_throughput']:.1f}  images/sec"
          f"  ({tp_stats['improvement_pct']:+.1f}% vs baseline)"
          f"  [effect: {tp_stats['practical_interpretation']}]")
    print(f"  Peak VRAM  : {flash_bench['mean_memory']:.3f} ± "
          f"{flash_bench['std_memory']:.3f}  GB"
          f"  ({mem_stats['improvement_pct']:+.1f}% reduction)")
    print(f"  Top-1 acc  : {flash_acc['top1_accuracy']:.4f}"
          f"  ({flash_acc['correct']}/{flash_acc['total']})"
          f"  ({acc_stats['diff_pct']:+.2f} pp vs baseline)")
    print(f"  Top-5 acc  : {flash_acc.get('top5_accuracy', float('nan')):.4f}")
    print()
    print("Statistical significance (α = 0.05):")
    sig = lambda s, k="significant": "YES (p={:.4f})".format(s["p_value"]) if s[k] else "no  (p={:.4f})".format(s["p_value"])
    print(f"  Throughput : {sig(tp_stats)}  g={tp_stats['effect_size']:.3f} [{tp_stats['practical_interpretation']}]")
    print(f"  Memory     : {sig(mem_stats)}  g={mem_stats['effect_size']:.3f} [{mem_stats['practical_interpretation']}]")
    print(f"  Accuracy   : {sig(acc_stats)}")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Save full stats to JSON
    # ------------------------------------------------------------------
    phase3_stats = {
        "experiment": "flash_only_vs_baseline",
        "label":      LABEL,
        "alpha":      0.05,
        "throughput": {
            **tp_stats,
            "baseline_mean": baseline_bench["mean_throughput"],
            "baseline_std":  baseline_bench["std_throughput"],
            "flash_mean":    flash_bench["mean_throughput"],
            "flash_std":     flash_bench["std_throughput"],
        },
        "memory": {
            **mem_stats,
            "baseline_mean": baseline_bench["mean_memory"],
            "baseline_std":  baseline_bench["std_memory"],
            "flash_mean":    flash_bench["mean_memory"],
            "flash_std":     flash_bench["std_memory"],
        },
        "accuracy": {
            **acc_stats,
            "baseline_top1": baseline_acc["top1_accuracy"],
            "flash_top1":    flash_acc["top1_accuracy"],
            "flash_top5":    flash_acc.get("top5_accuracy"),
        },
        "metadata": meta,
    }

    stats_path = os.path.join(RESULTS_DIR, "phase3_stats.json")
    with open(stats_path, "w") as fh:
        json.dump(phase3_stats, fh, indent=2)
    print(f"FlashAttention stats saved to {stats_path}")
    print(f"\nFlashAttention benchmark complete. Results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
