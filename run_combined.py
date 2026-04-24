"""
run_combined.py

Phase 4: ToMe + FlashAttention-2 combined DeiT-B/16 benchmark on ImageNet-1K val.

Runs throughput + peak-VRAM benchmark and top-1 accuracy evaluation for the
combined condition, then performs pairwise statistical comparisons against all
three prior conditions (baseline, ToMe-only r=8, FA-only) and prints a
consolidated four-condition results table.

Results saved to:
  results/combined_benchmark.json
  results/combined_accuracy.json
  results/phase4_stats.json

Usage:
    python run_combined.py [--batch_size 256] [--num_workers 8] [--fast] [--device cuda]

Flags:
    --fast     Smoke-test mode: only 10 batches for benchmark and accuracy eval.
    --device   Override compute device (default: cuda).
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

PROJECT_ROOT: str = "/scratch/rsp9219/project"
RESULTS_DIR: str  = os.path.join(PROJECT_ROOT, "results")

LABEL = "combined"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: ToMe + FlashAttention-2 combined DeiT-B/16 benchmark"
    )
    parser.add_argument("--batch_size",  type=int,  default=256,    help="Batch size (default: 256)")
    parser.add_argument("--num_workers", type=int,  default=8,      help="DataLoader workers (default: 8)")
    parser.add_argument("--fast",        action="store_true",       help="Smoke-test with only 10 batches")
    parser.add_argument("--device",      type=str,  default="cuda", help="Compute device (default: cuda)")
    parser.add_argument("--r",           type=int,  default=8,      help="ToMe merge tokens per block (default: 8)")
    parser.add_argument("--n_warmup",    type=int,  default=100,    help="Warm-up batches (default: 100)")
    parser.add_argument("--n_trials",    type=int,  default=100,    help="Timed trial batches (default: 100)")
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


def _load_json_pair(
    results_dir: str,
    label: str,
    description: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load benchmark + accuracy JSONs for a given label.

    Args:
        results_dir:  Directory containing result JSONs.
        label:        File stem, e.g. 'baseline' -> baseline_benchmark.json.
        description:  Human-readable name for error messages.

    Returns:
        (bench_dict, acc_dict)

    Raises:
        FileNotFoundError: If either JSON is missing.
    """
    bench_path = os.path.join(results_dir, f"{label}_benchmark.json")
    acc_path   = os.path.join(results_dir, f"{label}_accuracy.json")

    for p in (bench_path, acc_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{description} result not found: {p}\n"
                f"Run the corresponding phase script first."
            )

    with open(bench_path) as fh:
        bench = json.load(fh)
    with open(acc_path) as fh:
        acc = json.load(fh)

    return bench, acc


def load_prior_results(
    results_dir: str,
) -> Tuple[
    Dict[str, Any], Dict[str, Any],   # baseline
    Dict[str, Any], Dict[str, Any],   # tome_only
    Dict[str, Any], Dict[str, Any],   # flash_only
]:
    """Load all three prior-phase result pairs.

    ToMe-only results are tried under 'tome_r8' first (sweep naming),
    then 'tome_only' as a fallback.

    Returns:
        (baseline_bench, baseline_acc,
         tome_bench,     tome_acc,
         flash_bench,    flash_acc)
    """
    baseline_bench, baseline_acc = _load_json_pair(
        results_dir, "baseline", "Phase 1 baseline"
    )

    # ToMe-only: sweep scripts typically write 'tome_r8'; single-run scripts 'tome_only'
    tome_bench: Optional[Dict] = None
    tome_acc:   Optional[Dict] = None
    for tome_label in ("tome_r8", "tome_only"):
        try:
            tome_bench, tome_acc = _load_json_pair(results_dir, tome_label, "ToMe-only r=8")
            print(f"Loaded ToMe-only results from: {tome_label}_*.json")
            break
        except FileNotFoundError:
            continue
    if tome_bench is None:
        raise FileNotFoundError(
            f"ToMe-only results not found in {results_dir} "
            "(tried: tome_r8_benchmark.json, tome_only_benchmark.json).\n"
            "Run run_token_merging.py (or an equivalent single-run script) first."
        )

    flash_bench, flash_acc = _load_json_pair(
        results_dir, "flash_only", "Phase 3 FA-only"
    )

    print("Loaded prior-phase results:")
    for name, b, a in (
        ("Baseline",   baseline_bench, baseline_acc),
        ("ToMe r=8",   tome_bench,     tome_acc),
        ("FA-only",    flash_bench,    flash_acc),
    ):
        print(
            f"  {name:<10}: {b['mean_throughput']:.1f} ± {b['std_throughput']:.1f} img/s  "
            f"| mem {b['mean_memory']:.3f} GB  "
            f"| top-1 {a['top1_accuracy']:.4f}"
        )
    print()

    return baseline_bench, baseline_acc, tome_bench, tome_acc, flash_bench, flash_acc


def print_consolidated_table(
    baseline_bench: Dict[str, Any],
    baseline_acc:   Dict[str, Any],
    tome_bench:     Dict[str, Any],
    tome_acc:       Dict[str, Any],
    flash_bench:    Dict[str, Any],
    flash_acc:      Dict[str, Any],
    combined_bench: Dict[str, Any],
    combined_acc:   Dict[str, Any],
    stats:          Dict[str, Any],
) -> None:
    """Print a single consolidated four-condition table with significance flags."""
    from stats import compare_throughput, compare_memory, compare_accuracy

    W = 100
    print()
    print("=" * W)
    print("Phase 4 — Consolidated Results: Baseline | ToMe r=8 | FA-only | Combined")
    print("=" * W)

    # Column headers
    print(
        f"  {'Condition':<14}  {'Throughput (img/s)':>20}  "
        f"{'Peak VRAM (GB)':>16}  {'Top-1 Acc':>10}  {'vs Baseline':>12}"
    )
    print("-" * W)

    rows = [
        ("Baseline",    baseline_bench, baseline_acc),
        ("ToMe r=8",    tome_bench,     tome_acc),
        ("FA-only",     flash_bench,    flash_acc),
        ("Combined",    combined_bench, combined_acc),
    ]

    for name, bench, acc in rows:
        tp  = f"{bench['mean_throughput']:>7.1f} ± {bench['std_throughput']:.1f}"
        mem = f"{bench['mean_memory']:>6.3f} ± {bench['std_memory']:.3f}"
        top1 = f"{acc['top1_accuracy']:.4f}"

        if name == "Baseline":
            vs = "—"
        else:
            tp_s  = compare_throughput(baseline_bench, bench)
            sig   = "*" if tp_s["significant"] else " "
            vs    = f"{tp_s['improvement_pct']:+.1f}%{sig}"

        print(f"  {name:<14}  {tp:>20}  {mem:>16}  {top1:>10}  {vs:>12}")

    print("-" * W)
    print("  * = throughput improvement vs baseline is statistically significant (p < 0.05, Welch t-test)")
    print("=" * W)
    print()

    # Pairwise significance summary
    print("Pairwise significance (α = 0.05):")
    comparisons = [
        ("Combined vs Baseline", baseline_bench, baseline_acc, combined_bench, combined_acc),
        ("Combined vs ToMe r=8", tome_bench,     tome_acc,     combined_bench, combined_acc),
        ("Combined vs FA-only",  flash_bench,    flash_acc,    combined_bench, combined_acc),
    ]
    for desc, b_a, a_a, b_b, a_b in comparisons:
        tp_s  = compare_throughput(b_a, b_b)
        mem_s = compare_memory(b_a, b_b)
        acc_s = compare_accuracy(a_a, a_b)
        sig_tp  = f"p={tp_s['p_value']:.4f}{'*' if tp_s['significant'] else ''}"
        sig_mem = f"p={mem_s['p_value']:.4f}{'*' if mem_s['significant'] else ''}"
        sig_acc = f"p={acc_s['p_value']:.4f}{'*' if acc_s['significant'] else ''}"
        print(
            f"  {desc:<28}  tp: {sig_tp:<12}  "
            f"mem: {sig_mem:<12}  acc: {sig_acc}"
        )
    print()


def main() -> None:
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print_env_info(args.device)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    from models.combined import load_combined_model

    print(f"Loading combined model (DeiT-B/16 + ToMe r={args.r} + FlashAttention-2)...")
    model = load_combined_model(device=args.device, r=args.r)
    print()

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    from data.imagenet_loader import get_imagenet_val_loader, get_fast_loader

    if args.fast:
        print(f"[FAST MODE] Using only 10 batches (batch_size={args.batch_size}).")
        n_warmup = 3
        n_trials = 10
        bench_loader = get_fast_loader(
            max_batches=10,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        acc_loader = get_fast_loader(
            max_batches=10,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        n_warmup = args.n_warmup
        n_trials = args.n_trials
        bench_loader = get_imagenet_val_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        acc_loader = get_imagenet_val_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    print(
        f"DataLoader: {len(bench_loader)} batches, "
        f"batch_size={args.batch_size}, num_workers={args.num_workers}\n"
    )

    # ------------------------------------------------------------------
    # Benchmark (throughput + peak VRAM)
    # ------------------------------------------------------------------
    from benchmark import run_benchmark

    combined_bench = run_benchmark(
        model=model,
        loader=bench_loader,
        device=args.device,
        n_warmup=n_warmup,
        n_trials=n_trials,
        label=LABEL,
    )

    # ------------------------------------------------------------------
    # Accuracy evaluation
    # ------------------------------------------------------------------
    from evaluate import run_accuracy_eval

    combined_acc = run_accuracy_eval(
        model=model,
        loader=acc_loader,
        device=args.device,
        label=LABEL,
    )

    # ------------------------------------------------------------------
    # Load prior results and run pairwise comparisons
    # ------------------------------------------------------------------
    baseline_bench, baseline_acc, tome_bench, tome_acc, flash_bench, flash_acc = (
        load_prior_results(RESULTS_DIR)
    )

    from stats import compare_throughput, compare_memory, compare_accuracy, print_stats_table

    # Primary: Combined vs Baseline
    print_stats_table(baseline_bench, baseline_acc, combined_bench, combined_acc, LABEL)

    # Is FA2 adding on top of ToMe alone?
    print_stats_table(tome_bench, tome_acc, combined_bench, combined_acc, f"{LABEL} vs tome_r8")

    # Is ToMe adding on top of FA2 alone?
    print_stats_table(flash_bench, flash_acc, combined_bench, combined_acc, f"{LABEL} vs flash_only")

    # ------------------------------------------------------------------
    # Consolidated four-condition table
    # ------------------------------------------------------------------
    tp_vs_base  = compare_throughput(baseline_bench, combined_bench)
    mem_vs_base = compare_memory(baseline_bench, combined_bench)
    acc_vs_base = compare_accuracy(baseline_acc, combined_acc)

    tp_vs_tome  = compare_throughput(tome_bench, combined_bench)
    mem_vs_tome = compare_memory(tome_bench, combined_bench)
    acc_vs_tome = compare_accuracy(tome_acc, combined_acc)

    tp_vs_flash  = compare_throughput(flash_bench, combined_bench)
    mem_vs_flash = compare_memory(flash_bench, combined_bench)
    acc_vs_flash = compare_accuracy(flash_acc, combined_acc)

    phase4_stats = {
        "experiment": "combined_vs_all",
        "label":      LABEL,
        "r":          args.r,
        "alpha":      0.05,
        "combined_vs_baseline": {
            "throughput": {
                **tp_vs_base,
                "baseline_mean": baseline_bench["mean_throughput"],
                "combined_mean": combined_bench["mean_throughput"],
            },
            "memory": {
                **mem_vs_base,
                "baseline_mean": baseline_bench["mean_memory"],
                "combined_mean": combined_bench["mean_memory"],
            },
            "accuracy": {
                **acc_vs_base,
                "baseline_top1": baseline_acc["top1_accuracy"],
                "combined_top1": combined_acc["top1_accuracy"],
            },
        },
        "combined_vs_tome_r8": {
            "throughput": {**tp_vs_tome},
            "memory":     {**mem_vs_tome},
            "accuracy":   {**acc_vs_tome},
        },
        "combined_vs_flash_only": {
            "throughput": {**tp_vs_flash},
            "memory":     {**mem_vs_flash},
            "accuracy":   {**acc_vs_flash},
        },
    }

    stats_path = os.path.join(RESULTS_DIR, "phase4_stats.json")
    with open(stats_path, "w") as fh:
        json.dump(phase4_stats, fh, indent=2)
    print(f"Phase 4 stats saved to {stats_path}")

    # ------------------------------------------------------------------
    # Consolidated print
    # ------------------------------------------------------------------
    print_consolidated_table(
        baseline_bench, baseline_acc,
        tome_bench,     tome_acc,
        flash_bench,    flash_acc,
        combined_bench, combined_acc,
        phase4_stats,
    )

    # ------------------------------------------------------------------
    # Phase 4 summary
    # ------------------------------------------------------------------
    sig = lambda s, k="significant": (
        f"YES (p={s['p_value']:.4f})" if s[k] else f"no  (p={s['p_value']:.4f})"
    )

    print("=" * 60)
    print("Phase 4 — Combined ToMe r=8 + FlashAttention-2 (DeiT-B/16)")
    print("=" * 60)
    print(
        f"  Throughput : {combined_bench['mean_throughput']:.1f} ± "
        f"{combined_bench['std_throughput']:.1f}  images/sec"
        f"  ({tp_vs_base['improvement_pct']:+.1f}% vs baseline)"
    )
    print(
        f"  Peak VRAM  : {combined_bench['mean_memory']:.3f} ± "
        f"{combined_bench['std_memory']:.3f}  GB"
        f"  ({mem_vs_base['improvement_pct']:+.1f}% reduction)"
    )
    print(
        f"  Top-1 acc  : {combined_acc['top1_accuracy']:.4f}"
        f"  ({combined_acc['correct']}/{combined_acc['total']})"
        f"  ({acc_vs_base['diff_pct']:+.2f} pp vs baseline)"
    )
    print()
    print("Statistical significance (α = 0.05, vs baseline):")
    print(f"  Throughput : {sig(tp_vs_base)}")
    print(f"  Memory     : {sig(mem_vs_base)}")
    print(f"  Accuracy   : {sig(acc_vs_base)}")
    print()
    print("Additive gains (combined vs each single technique):")
    print(
        f"  vs ToMe r=8  — throughput: {tp_vs_tome['improvement_pct']:+.1f}%  "
        f"{sig(tp_vs_tome)}"
    )
    print(
        f"  vs FA-only   — throughput: {tp_vs_flash['improvement_pct']:+.1f}%  "
        f"{sig(tp_vs_flash)}"
    )
    print("=" * 60)
    print(f"\nPhase 4 complete. Results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
