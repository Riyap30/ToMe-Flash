"""
Statistical comparison utilities for the ToMe-Flash benchmark (Phase 4).

Implements:
  compare_throughput()  — Welch's two-sample t-test on throughput lists
  compare_memory()      — Welch's two-sample t-test on peak-memory lists
  compare_accuracy()    — Two-proportion z-test on top-1 counts
  print_stats_table()   — Pretty comparison table with CIs and p-values

All tests use α = 0.05 (pre-registered). Confidence intervals are 95%.
"""

import math
from typing import Any, Dict, List

import numpy as np
import scipy.stats as sci_stats


def _welch_ci(
    a: List[float],
    b: List[float],
    alpha: float = 0.05,
) -> tuple:
    """Return the 95% CI on the difference of means (a - b) using Welch's t."""
    na, nb     = len(a), len(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b   = np.var(a, ddof=1), np.var(b, ddof=1)

    se = math.sqrt(var_a / na + var_b / nb)

    # Welch-Satterthwaite degrees of freedom
    num   = (var_a / na + var_b / nb) ** 2
    denom = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
    df    = num / denom if denom > 0 else na + nb - 2

    t_crit = sci_stats.t.ppf(1 - alpha / 2, df)
    diff   = mean_a - mean_b
    ci_low, ci_high = diff - t_crit * se, diff + t_crit * se
    return ci_low, ci_high


def compare_throughput(results_a: Dict[str, Any], results_b: Dict[str, Any]) -> Dict[str, Any]:
    """Welch's two-sample t-test on throughput (images/sec) lists.

    Args:
        results_a: Benchmark dict from run_benchmark() for condition A (baseline).
        results_b: Benchmark dict from run_benchmark() for condition B.

    Returns:
        dict with keys:
          t_stat, p_value, significant, ci_low, ci_high,
          mean_a, mean_b, improvement_pct
    """
    a: List[float] = results_a["throughputs"]
    b: List[float] = results_b["throughputs"]

    t_stat, p_value = sci_stats.ttest_ind(b, a, equal_var=False)
    ci_low, ci_high = _welch_ci(b, a)

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    improvement_pct = ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else float("nan")

    return {
        "t_stat":          float(t_stat),
        "p_value":         float(p_value),
        "significant":     bool(p_value < 0.05),
        "ci_low":          float(ci_low),
        "ci_high":         float(ci_high),
        "mean_a":          mean_a,
        "mean_b":          mean_b,
        "improvement_pct": float(improvement_pct),
    }


def compare_memory(results_a: Dict[str, Any], results_b: Dict[str, Any]) -> Dict[str, Any]:
    """Welch's two-sample t-test on peak GPU memory (GB) lists.

    Args:
        results_a: Benchmark dict from run_benchmark() for condition A (baseline).
        results_b: Benchmark dict from run_benchmark() for condition B.

    Returns:
        dict with same structure as compare_throughput() but for peak_memories.
        improvement_pct is positive when B uses *less* memory than A.
    """
    a: List[float] = results_a["peak_memories"]
    b: List[float] = results_b["peak_memories"]

    t_stat, p_value = sci_stats.ttest_ind(b, a, equal_var=False)
    ci_low, ci_high = _welch_ci(b, a)

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    # Positive improvement_pct = B uses less memory (reduction is good)
    improvement_pct = ((mean_a - mean_b) / mean_a * 100) if mean_a != 0 else float("nan")

    return {
        "t_stat":          float(t_stat),
        "p_value":         float(p_value),
        "significant":     bool(p_value < 0.05),
        "ci_low":          float(ci_low),
        "ci_high":         float(ci_high),
        "mean_a":          mean_a,
        "mean_b":          mean_b,
        "improvement_pct": float(improvement_pct),
    }


def compare_accuracy(acc_a: Dict[str, Any], acc_b: Dict[str, Any]) -> Dict[str, Any]:
    """Two-proportion z-test comparing top-1 accuracy between two conditions.

    Hypothesis test: pooled-SE z-test under H0: p_a == p_b (two-sided).
    Confidence interval: standard two-proportion Z-interval using unpooled SE,
    i.e. CI on (p_b - p_a) = (p_b - p_a) ± z * sqrt(p_b(1-p_b)/n_b + p_a(1-p_a)/n_a).

    Both computed with scipy.stats.norm — no external dependencies.

    Args:
        acc_a: Accuracy dict from run_accuracy_eval() for condition A (baseline).
        acc_b: Accuracy dict from run_accuracy_eval() for condition B.

    Returns:
        dict with keys:
          z_stat, p_value, significant, ci_low, ci_high, prop_a, prop_b, diff_pct
    """
    n_a, n_b      = acc_a["total"],   acc_b["total"]
    k_a, k_b      = acc_a["correct"], acc_b["correct"]
    prop_a: float = acc_a["top1_proportion"]
    prop_b: float = acc_b["top1_proportion"]

    # Pooled SE for hypothesis test (H0: p_a == p_b)
    p_pool  = (k_a + k_b) / (n_a + n_b)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_stat  = (prop_b - prop_a) / se_pool if se_pool > 0 else 0.0
    p_value = float(2 * (1 - sci_stats.norm.cdf(abs(z_stat))))

    diff_pct: float = (prop_b - prop_a) * 100

    # Standard two-proportion Z-interval: unpooled SE for CI estimation
    z_crit  = sci_stats.norm.ppf(0.975)  # 1.96 for 95% CI
    se_diff = math.sqrt(prop_b * (1 - prop_b) / n_b + prop_a * (1 - prop_a) / n_a)
    ci_low  = float(diff_pct - z_crit * se_diff * 100)
    ci_high = float(diff_pct + z_crit * se_diff * 100)

    return {
        "z_stat":     float(z_stat),
        "p_value":    float(p_value),
        "significant": bool(p_value < 0.05),
        "ci_low":     ci_low,
        "ci_high":    ci_high,
        "prop_a":     prop_a,
        "prop_b":     prop_b,
        "diff_pct":   diff_pct,
    }


def print_stats_table(
    baseline_bench: Dict[str, Any],
    baseline_acc:   Dict[str, Any],
    other_bench:    Dict[str, Any],
    other_acc:      Dict[str, Any],
    label:          str,
) -> None:
    """Pretty-print a comparison table vs baseline.

    Prints means, 95% CIs, t/z statistics, p-values, and significance flags.

    Args:
        baseline_bench: run_benchmark() result for baseline condition.
        baseline_acc:   run_accuracy_eval() result for baseline condition.
        other_bench:    run_benchmark() result for the comparison condition.
        other_acc:      run_accuracy_eval() result for the comparison condition.
        label:          Name of the comparison condition (e.g. 'tome_r8').
    """
    tp_stats  = compare_throughput(baseline_bench, other_bench)
    mem_stats = compare_memory(baseline_bench, other_bench)
    acc_stats = compare_accuracy(baseline_acc, other_acc)

    sig = lambda s: "YES *" if s["significant"] else "no"

    W = 70
    print("=" * W)
    print(f"Statistical comparison: baseline  vs  {label}")
    print("=" * W)

    # --- Throughput ---
    print(f"\n{'Throughput (images/sec)':}")
    print(f"  Baseline : {tp_stats['mean_a']:.1f}")
    print(f"  {label:<12}: {tp_stats['mean_b']:.1f}  ({tp_stats['improvement_pct']:+.1f}%)")
    print(f"  95% CI on diff : [{tp_stats['ci_low']:.1f}, {tp_stats['ci_high']:.1f}]")
    print(f"  Welch t={tp_stats['t_stat']:.3f}  p={tp_stats['p_value']:.4f}  significant: {sig(tp_stats)}")

    # --- Memory ---
    print(f"\n{'Peak GPU memory (GB)':}")
    print(f"  Baseline : {mem_stats['mean_a']:.3f}")
    print(f"  {label:<12}: {mem_stats['mean_b']:.3f}  ({mem_stats['improvement_pct']:+.1f}% reduction)")
    print(f"  95% CI on diff : [{mem_stats['ci_low']:.3f}, {mem_stats['ci_high']:.3f}]")
    print(f"  Welch t={mem_stats['t_stat']:.3f}  p={mem_stats['p_value']:.4f}  significant: {sig(mem_stats)}")

    # --- Accuracy ---
    print(f"\n{'Top-1 accuracy':}")
    print(f"  Baseline : {acc_stats['prop_a']:.4f}")
    print(f"  {label:<12}: {acc_stats['prop_b']:.4f}  ({acc_stats['diff_pct']:+.2f} pp)")
    print(f"  95% CI on diff : [{acc_stats['ci_low']:.4f}, {acc_stats['ci_high']:.4f}]")
    print(f"  z={acc_stats['z_stat']:.3f}  p={acc_stats['p_value']:.4f}  significant: {sig(acc_stats)}")

    print("\n" + "=" * W)
    print("Note: α = 0.05 (pre-registered). Throughput/memory: Welch's t-test.")
    print("Accuracy: two-proportion z-test (statsmodels).")
    print("=" * W + "\n")


def summarize_tome_sweep(sweep_results: list) -> None:
    """Print an ASCII summary table for the full ToMe r-value sweep.

    Args:
        sweep_results: The 'results' list from tome_sweep_summary.json.
                       Each entry must have the keys written by run_token_merging.py.
    """
    # Identify recommended_r: largest r with accuracy_drop < 1% and significant
    qualifying = [
        e for e in sweep_results
        if e.get("accuracy_drop_pct") is not None
        and e["accuracy_drop_pct"] < 1.0
        and e["significant"]
    ]
    if qualifying:
        recommended_r = max(qualifying, key=lambda e: e["r"])["r"]
    elif sweep_results:
        valid = [e for e in sweep_results if e.get("accuracy_drop_pct") is not None]
        recommended_r = (
            min(valid, key=lambda e: e["accuracy_drop_pct"])["r"] if valid
            else sweep_results[0]["r"]
        )
    else:
        recommended_r = None

    col_w = 76
    header = (
        f"  {'r':>2}  {'tokens':>6}  {'thrpt (img/s)':>13}  "
        f"{'vs base':>8}  {'p-val':>7}  {'sig':>3}  {'top-1':>6}  {'acc drop':>9}"
    )
    sep = "-" * col_w

    print()
    print("=" * col_w)
    print("Token Merging Sweep — Summary Table")
    print("=" * col_w)
    print(header)
    print(sep)

    for e in sweep_results:
        marker = "→" if e["r"] == recommended_r else "  "
        sig    = " * " if e["significant"] else "   "
        thrpt  = f"{e['mean_throughput']:.1f} ± {e['std_throughput']:.3f}"
        vs_base = f"{e['throughput_improvement_pct']:+.1f}%"

        if e.get("top1_accuracy") is not None:
            acc  = f"{e['top1_accuracy']:.4f}"
            drop = f"{e['accuracy_drop_pct']:+.2f}%"
        else:
            acc  = "  N/A "
            drop = "   N/A "

        print(
            f"{marker} {e['r']:>2}  {e['final_tokens']:>6}  "
            f"{thrpt:>13}  {vs_base:>8}  "
            f"{e['p_value']:>7.4f}  {sig}  {acc}  {drop}"
        )

    print(sep)
    print(f"→ = recommended r  |  * = p < 0.05  |  α = 0.05 (pre-registered)")
    if recommended_r is not None:
        print(f"Recommended r = {recommended_r}  (use for combined ToMe+FlashAttn condition)")
    print("=" * col_w)
    print()
