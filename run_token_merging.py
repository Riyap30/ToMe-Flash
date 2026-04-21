"""
run_token_merging.py

Entry point for the Token Merging (ToMe) sweep over r ∈ {4, 8, 16}.

For each r value this script:
  1. Loads a fresh DeiT-B/16 patched with tome.patch.timm(model, r=r)
  2. Runs run_benchmark() — 30 warm-up + 30 timed trials
  3. Runs run_accuracy_eval() over the full 50 K ImageNet-1K val set
  4. Runs Welch's t-test and two-proportion z-test vs the validated baseline
  5. Prints per-r statistics table

After all r values, prints a combined summary table and writes
results/tome_sweep_summary.json.

Usage:
    python run_token_merging.py [--r_values 4 8 16] [--batch_size 64]
                                [--num_workers 8] [--fast] [--device cuda]
                                [--skip_accuracy]
"""

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT: str = "/scratch/rsp9219/project"
RESULTS_DIR: str  = os.path.join(PROJECT_ROOT, "results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ToMe sweep: benchmark DeiT-B/16 with r ∈ {4,8,16}"
    )
    parser.add_argument(
        "--r_values", type=int, nargs="+", default=[4, 8, 12, 14, 16],
        help="Merge ratios to sweep (default: 4 8 12 14 16)",
    )
    parser.add_argument("--batch_size",    type=int,  default=64,     help="Batch size (default: 64)")
    parser.add_argument("--num_workers",   type=int,  default=8,      help="DataLoader workers (default: 8)")
    parser.add_argument("--fast",          action="store_true",       help="Smoke-test with only 10 batches")
    parser.add_argument("--device",        type=str,  default="cuda", help="Compute device (default: cuda)")
    parser.add_argument("--n_warmup",      type=int,  default=100,    help="Warm-up batches per condition (default: 100)")
    parser.add_argument("--n_trials",      type=int,  default=100,    help="Timed trial batches per condition (default: 100)")
    parser.add_argument("--skip_accuracy", action="store_true",       help="Skip 50K accuracy eval")
    parser.add_argument("--skip_r0_check", action="store_true",       help="Skip r=0 sanity check")
    return parser.parse_args()


def print_env_info(device: str) -> None:
    print("=" * 60)
    print("Environment")
    print("=" * 60)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.version.cuda}")
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"  GPU     : {torch.cuda.get_device_name(idx)}")
        total = torch.cuda.get_device_properties(idx).total_memory / 1e9
        print(f"  VRAM    : {total:.1f} GB")
    else:
        print("  GPU     : (not available)")
    print("=" * 60)
    print()


def load_baseline_results(results_dir: str) -> tuple:
    """Load validated baseline JSON results.

    Returns:
        (baseline_bench dict, baseline_acc dict)

    Raises:
        FileNotFoundError: If baseline JSON files are missing.
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

    print("Loaded baseline results:")
    print(f"  Throughput : {baseline_bench['mean_throughput']:.3f} ± "
          f"{baseline_bench['std_throughput']:.3f} img/sec")
    print(f"  Memory     : {baseline_bench['mean_memory']:.4f} ± "
          f"{baseline_bench['std_memory']:.4f} GB")
    print(f"  Top-1 acc  : {baseline_acc['top1_accuracy']:.5f} "
          f"({baseline_acc['correct']}/{baseline_acc['total']})")
    print()
    return baseline_bench, baseline_acc


def _select_recommended_r(sweep_entries: list) -> int:
    """Return the recommended r value.

    Rule: largest r where accuracy_drop_pct < 1.0 AND throughput improvement
    is statistically significant (p < 0.05).
    Fallback: r with minimum accuracy drop if none satisfy the accuracy constraint.
    """
    qualifying = [
        e for e in sweep_entries
        if e["accuracy_drop_pct"] < 1.0 and e["significant"]
    ]
    if qualifying:
        return max(qualifying, key=lambda e: e["r"])["r"]
    # fallback
    return min(sweep_entries, key=lambda e: e["accuracy_drop_pct"])["r"]


def print_final_summary(sweep_entries: list, recommended_r: int) -> None:
    """Print a combined summary table for all r values."""
    header = (
        f"{'':>2}  {'r':>2}  {'tokens':>6}  {'thrpt':>8}  {'±':>6}  "
        f"{'vs base':>8}  {'p-val':>7}  {'sig':>3}  {'top-1':>6}  {'acc drop':>9}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("ToMe Sweep — Final Summary")
    print("=" * len(header))
    print(header)
    print(sep)
    for e in sweep_entries:
        marker = "→" if e["r"] == recommended_r else "  "
        sig    = " * " if e["significant"] else "   "
        acc    = f"{e['top1_accuracy']:.4f}" if e["top1_accuracy"] is not None else "  N/A "
        drop   = f"{e['accuracy_drop_pct']:+.2f}%" if e["accuracy_drop_pct"] is not None else "   N/A "
        print(
            f"{marker}  {e['r']:>2}  {e['final_tokens']:>6}  "
            f"{e['mean_throughput']:>8.1f}  {e['std_throughput']:>6.3f}  "
            f"{e['throughput_improvement_pct']:>+7.1f}%  "
            f"{e['p_value']:>7.4f}  {sig}  {acc}  {drop}"
        )
    print(sep)
    print(f"→ = recommended r  |  * = p < 0.05  |  α = 0.05 (pre-registered)")
    print(f"Recommended r: {recommended_r}")
    print("=" * len(header))
    print()


def run_r0_sanity_check(args: argparse.Namespace, baseline_bench: dict) -> None:
    """Verify that ToMe with r=0 is numerically equivalent to the unpatched baseline.

    Runs three checks:
      1. Logit agreement  — same batch, max absolute diff must be < 1e-2
      2. Throughput parity — r=0 throughput within 5% of saved baseline
      3. Accuracy parity  — top-1 on a 20-batch mini-eval within 0.1% of baseline

    Raises AssertionError with a descriptive message if any check fails, which
    causes the caller to abort before the main r-value sweep.

    Args:
        args:           Parsed CLI arguments (device, batch_size, num_workers).
        baseline_bench: Loaded baseline benchmark dict (keys: mean_throughput, …).
    """
    from models.baseline import load_baseline_model
    from models.tome_only import load_tome_model
    from data.imagenet_loader import get_fast_loader
    from benchmark import run_benchmark
    from evaluate import run_accuracy_eval

    print()
    print("=" * 60)
    print("r=0 Sanity Check  (ToMe patch must not alter outputs at r=0)")
    print("=" * 60)

    model_base = load_baseline_model(device=args.device)
    model_r0   = load_tome_model(r=0, device=args.device)

    # ------------------------------------------------------------------ 1. logits
    single_loader = get_fast_loader(
        max_batches=1, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    images, _ = next(iter(single_loader))
    images = images.to(args.device, dtype=torch.bfloat16, non_blocking=True)

    with torch.no_grad():
        logits_base = model_base(images)
        logits_r0   = model_r0(images)

    max_abs_diff = (logits_base - logits_r0).abs().max().item()
    assert torch.allclose(logits_base, logits_r0, rtol=1e-2, atol=1e-2), (
        f"SANITY FAIL [logits]: r=0 logits diverge from unpatched baseline. "
        f"Max absolute difference: {max_abs_diff:.6f} (tolerance 1e-2). "
        "The ToMe patch introduces numerical error even at r=0 — "
        "all ToMe results are suspect. Aborting sweep."
    )
    print(f"  [PASS] Logit check        : max |diff| = {max_abs_diff:.2e}  (tol 1e-2)")

    # ------------------------------------------------------------------ 2. throughput
    n_warmup_sc = 5
    n_trials_sc = 10
    bench_loader = get_fast_loader(
        max_batches=20, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    r0_bench = run_benchmark(
        model=model_r0,
        loader=bench_loader,
        device=args.device,
        n_warmup=n_warmup_sc,
        n_trials=n_trials_sc,
        label="sanity_r0_bench",
    )

    baseline_tp = baseline_bench["mean_throughput"]
    r0_tp       = r0_bench["mean_throughput"]
    tp_deviation = abs(r0_tp - baseline_tp) / baseline_tp

    assert tp_deviation <= 0.05, (
        f"SANITY FAIL [throughput]: r=0 throughput ({r0_tp:.1f} img/s) deviates "
        f"{tp_deviation:.1%} from baseline ({baseline_tp:.1f} img/s), exceeding the "
        "5% tolerance. Unexpected overhead from the ToMe patch at r=0. Aborting sweep."
    )
    print(
        f"  [PASS] Throughput check   : r=0 = {r0_tp:.1f}  baseline = {baseline_tp:.1f}"
        f"  deviation = {tp_deviation:.1%}  (tol 5%)"
    )

    # ------------------------------------------------------------------ 3. accuracy
    mini_batches = 20
    acc_loader_base = get_fast_loader(
        max_batches=mini_batches, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    acc_loader_r0 = get_fast_loader(
        max_batches=mini_batches, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    base_acc = run_accuracy_eval(
        model=model_base, loader=acc_loader_base,
        device=args.device, label="sanity_base_acc",
    )
    r0_acc = run_accuracy_eval(
        model=model_r0, loader=acc_loader_r0,
        device=args.device, label="sanity_r0_acc",
    )

    acc_diff = abs(base_acc["top1_accuracy"] - r0_acc["top1_accuracy"])
    assert acc_diff <= 0.001, (
        f"SANITY FAIL [accuracy]: r=0 top-1 ({r0_acc['top1_accuracy']:.4f}) differs "
        f"from baseline ({base_acc['top1_accuracy']:.4f}) by {acc_diff:.4f}, "
        "exceeding the 0.1% tolerance. The ToMe patch changes predictions at r=0. "
        "Aborting sweep."
    )
    print(
        f"  [PASS] Accuracy check     : r=0 = {r0_acc['top1_accuracy']:.4f}"
        f"  baseline = {base_acc['top1_accuracy']:.4f}"
        f"  |diff| = {acc_diff:.4f}  (tol 0.001)"
    )

    print()
    print("r=0 sanity check PASSED — ToMe patch is transparent at r=0.")
    print("=" * 60)
    print()

    del model_base, model_r0
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print_env_info(args.device)

    # ------------------------------------------------------------------ baseline
    baseline_bench, baseline_acc = load_baseline_results(RESULTS_DIR)

    # ------------------------------------------------------------------ r=0 sanity check
    if not args.skip_r0_check:
        run_r0_sanity_check(args, baseline_bench)
    else:
        print("[skip_r0_check] Skipping r=0 sanity check.")

    # ------------------------------------------------------------------ data loader
    from data.imagenet_loader import get_imagenet_val_loader, get_fast_loader
    from benchmark import run_benchmark
    from evaluate import run_accuracy_eval
    from stats import compare_throughput, compare_memory, compare_accuracy, print_stats_table

    if args.fast:
        print("[FAST MODE] Using only 10 batches per condition.")
        n_warmup, n_trials = 3, 10
    else:
        n_warmup, n_trials = args.n_warmup, args.n_trials

    print(f"Benchmark config: {n_warmup} warmup + {n_trials} timed trials per condition.")

    sweep_entries = []

    for r in args.r_values:
        print()
        print("*" * 60)
        print(f"  Token Merging  r = {r}")
        print("*" * 60)

        # ---- fresh model -----------------------------------------------
        from models.tome_only import load_tome_model, get_tome_info

        model = load_tome_model(r=r, device=args.device)
        get_tome_info(model, r)

        # ---- benchmark -------------------------------------------------
        if args.fast:
            bench_loader = get_fast_loader(
                max_batches=10, batch_size=args.batch_size, num_workers=args.num_workers,
            )
        else:
            bench_loader = get_imagenet_val_loader(
                batch_size=args.batch_size, num_workers=args.num_workers,
            )

        tome_bench = run_benchmark(
            model=model,
            loader=bench_loader,
            device=args.device,
            n_warmup=n_warmup,
            n_trials=n_trials,
            label=f"tome_r{r}",
        )

        # ---- accuracy --------------------------------------------------
        tome_acc = None
        if not args.skip_accuracy:
            if args.fast:
                acc_loader = get_fast_loader(
                    max_batches=10, batch_size=args.batch_size, num_workers=args.num_workers,
                )
            else:
                acc_loader = get_imagenet_val_loader(
                    batch_size=args.batch_size, num_workers=args.num_workers,
                )
            tome_acc = run_accuracy_eval(
                model=model,
                loader=acc_loader,
                device=args.device,
                label=f"tome_r{r}",
            )

        # ---- statistics ------------------------------------------------
        tp_stats  = compare_throughput(baseline_bench, tome_bench)
        mem_stats = compare_memory(baseline_bench, tome_bench)

        if tome_acc is not None:
            acc_stats = compare_accuracy(baseline_acc, tome_acc)
        else:
            acc_stats = None

        # Print per-condition stats table
        if tome_acc is not None:
            print_stats_table(baseline_bench, baseline_acc, tome_bench, tome_acc, f"tome_r{r}")
        else:
            # Throughput + memory only
            _print_partial_stats(tp_stats, mem_stats, r)

        # Statistical significance vs practical significance note
        abs_gain = tp_stats["improvement_pct"]
        if tp_stats["significant"] and abs_gain < 3.0:
            print(
                f"  [NOTE] Throughput gain for r={r} is statistically significant "
                f"(p={tp_stats['p_value']:.4f}) but modest ({abs_gain:+.1f}%). "
                "Consider practical significance alongside the p-value."
            )

        # ---- collect sweep entry ----------------------------------------
        acc_drop = None
        z_stat   = None
        acc_pval = None
        top1     = None
        if tome_acc is not None and acc_stats is not None:
            top1     = tome_acc["top1_accuracy"]
            acc_drop = round((baseline_acc["top1_accuracy"] - top1) * 100, 4)
            z_stat   = acc_stats["z_stat"]
            acc_pval = acc_stats["p_value"]

        sweep_entries.append({
            "r":                         r,
            "final_tokens":              197 - r * 12,
            "reduction_pct":             round((r * 12 / 197) * 100, 1),
            "mean_throughput":           tome_bench["mean_throughput"],
            "std_throughput":            tome_bench["std_throughput"],
            "throughput_improvement_pct": tp_stats["improvement_pct"],
            "t_stat":                    tp_stats["t_stat"],
            "p_value":                   tp_stats["p_value"],
            "significant":               tp_stats["significant"],
            "top1_accuracy":             top1,
            "accuracy_drop_pct":         acc_drop,
            "z_stat":                    z_stat,
            "acc_p_value":               acc_pval,
        })

        # Free GPU memory before next model load
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ recommended r
    # For recommended_r logic we need accuracy_drop_pct; if skipped, default to r=4
    if not args.skip_accuracy:
        recommended_r = _select_recommended_r(sweep_entries)
    else:
        recommended_r = args.r_values[0]
        print("[skip_accuracy] Cannot compute recommended_r — defaulting to smallest r.")

    # ------------------------------------------------------------------ final table
    print_final_summary(sweep_entries, recommended_r)

    # ------------------------------------------------------------------ sweep summary JSON
    summary = {
        "experiment":          "token_merging_sweep",
        "alpha":               0.05,
        "baseline_throughput": baseline_bench["mean_throughput"],
        "baseline_accuracy":   baseline_acc["top1_accuracy"],
        "results":             sweep_entries,
        "recommended_r":       recommended_r,
    }
    summary_path = os.path.join(RESULTS_DIR, "tome_sweep_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Sweep summary saved to {summary_path}")
    print(f"Recommended r = {recommended_r}  (used for combined ToMe+FlashAttn condition)")


def _print_partial_stats(tp_stats: dict, mem_stats: dict, r: int) -> None:
    """Print throughput+memory stats when accuracy eval was skipped."""
    W = 60
    print("=" * W)
    print(f"Stats vs baseline  (tome_r{r}, accuracy skipped)")
    print("=" * W)
    sig = lambda s: "YES *" if s["significant"] else "no"
    print(f"  Throughput  baseline : {tp_stats['mean_a']:.1f} img/sec")
    print(f"  Throughput  tome_r{r} : {tp_stats['mean_b']:.1f}  ({tp_stats['improvement_pct']:+.1f}%)")
    print(f"  Welch t={tp_stats['t_stat']:.3f}  p={tp_stats['p_value']:.4f}  sig: {sig(tp_stats)}")
    print(f"  Memory      baseline : {mem_stats['mean_a']:.3f} GB")
    print(f"  Memory      tome_r{r} : {mem_stats['mean_b']:.3f}  ({mem_stats['improvement_pct']:+.1f}% reduction)")
    print(f"  Welch t={mem_stats['t_stat']:.3f}  p={mem_stats['p_value']:.4f}  sig: {sig(mem_stats)}")
    print("=" * W)


if __name__ == "__main__":
    main()
