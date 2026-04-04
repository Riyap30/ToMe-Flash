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

    Uses statsmodels.stats.proportion.proportions_ztest (two-sided).

    Args:
        acc_a: Accuracy dict from run_accuracy_eval() for condition A (baseline).
        acc_b: Accuracy dict from run_accuracy_eval() for condition B.

    Returns:
        dict with keys:
          z_stat, p_value, significant, ci_low, ci_high, prop_a, prop_b, diff_pct
    """
    from statsmodels.stats.proportion import proportions_ztest, proportion_confint

    count = [acc_b["correct"], acc_a["correct"]]
    nobs  = [acc_b["total"],   acc_a["total"]]

    z_stat, p_value = proportions_ztest(count, nobs, alternative="two-sided")

    prop_a: float = acc_a["top1_proportion"]
    prop_b: float = acc_b["top1_proportion"]
    diff_pct: float = (prop_b - prop_a) * 100

    # 95% Wilson CI on the difference (approximate: difference of individual CIs)
    ci_lo_a, ci_hi_a = proportion_confint(acc_a["correct"], acc_a["total"], alpha=0.05, method="wilson")
    ci_lo_b, ci_hi_b = proportion_confint(acc_b["correct"], acc_b["total"], alpha=0.05, method="wilson")
    # Conservative CI on difference: (b_lo - a_hi, b_hi - a_lo)
    ci_low  = float(ci_lo_b - ci_hi_a)
    ci_high = float(ci_hi_b - ci_lo_a)

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
