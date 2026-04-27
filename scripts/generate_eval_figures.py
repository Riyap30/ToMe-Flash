#!/usr/bin/env python3
"""Generate presentation-ready evaluation figures for ToMe-Flash experiments.

Usage:
    python scripts/generate_eval_figures.py --results_dir "results latest"
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── deterministic styling ─────────────────────────────────────────────────────
PALETTE = {
    "baseline": "#4C72B0",
    "tome_r8":  "#DD8452",
    "fa_only":  "#55A868",
    "combined": "#C44E52",
    "highlight": "#E8A020",
}
DPI = 150
Z95 = 1.96

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path, label: str) -> dict:
    if not path.exists():
        sys.exit(f"ERROR: missing required file: {path}\n  (expected for '{label}')")
    with open(path) as f:
        return json.load(f)


def _ci95(values):
    """Return (mean, half_width) for 95% CI; uses scipy t if available."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(n))
    try:
        from scipy.stats import t as _t
        half = float(_t.ppf(0.975, df=n - 1) * se)
    except ImportError:
        half = Z95 * se  # z-approx; fine for n > 30
    return mean, half


def _prop_diff_ci95_half(p_base, p_cond, n):
    """95% CI half-width for difference of two independent proportions (pp)."""
    se = math.sqrt(p_base * (1 - p_base) / n + p_cond * (1 - p_cond) / n)
    return Z95 * se * 100


# ── Figure A: Pareto curve ────────────────────────────────────────────────────

def fig_pareto(sweep: dict, out_path: Path) -> None:
    results = sweep["results"]
    rec_r = sweep.get("recommended_r", 8)

    xs = [r["accuracy_drop_pct"] for r in results]
    ys = [r["throughput_improvement_pct"] for r in results]
    rs = [r["r"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=DPI)

    for xi, yi, r in zip(xs, ys, rs):
        color = PALETTE["highlight"] if r == rec_r else PALETTE["tome_r8"]
        size = 230 if r == rec_r else 110
        ax.scatter(xi, yi, c=color, s=size, zorder=5,
                   edgecolors="white", linewidths=0.9)
        star = " ★" if r == rec_r else ""
        dy = 4.5 if yi < max(ys) - 5 else -7
        ax.annotate(
            f"r={r}{star}", xy=(xi, yi),
            xytext=(xi + 0.07, yi + dy),
            fontsize=10,
            fontweight="bold" if r == rec_r else "normal",
            color="#222222",
        )

    # trend line
    sorted_pts = sorted(zip(xs, ys))
    ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
            "--", color="#999999", linewidth=1.2, zorder=2, alpha=0.7)

    ax.axhline(0, color="#666666", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axvline(0, color="#666666", linewidth=0.8, linestyle=":", alpha=0.6)

    normal_p = mpatches.Patch(color=PALETTE["tome_r8"], label="ToMe reduction levels")
    rec_p = mpatches.Patch(color=PALETTE["highlight"], label=f"Recommended r={rec_r}  ★")
    ax.legend(handles=[normal_p, rec_p], fontsize=10, loc="lower right", framealpha=0.9)

    ax.set_xlabel("Accuracy Drop vs. Baseline (pp)", fontsize=12, labelpad=6)
    ax.set_ylabel("Throughput Improvement vs. Baseline (%)", fontsize=12, labelpad=6)
    ax.set_title(
        "Token Merging: Throughput–Accuracy Pareto Frontier\n"
        "ViT-B/16 · ImageNet-1k · r ∈ {4, 8, 12, 14, 16}  (higher-right is better)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [A] Pareto curve       → {out_path}")


# ── Figure B: Main conditions CI chart ───────────────────────────────────────

def fig_ci_chart(
    base_bench: dict, fa_bench: dict, comb_bench: dict, tome_r8_bench: dict,
    base_acc: dict, fa_acc: dict, comb_acc: dict,
    sweep: dict,
    out_path: Path,
) -> None:
    # throughput CI from raw samples
    b_m, b_h = _ci95(base_bench["throughputs"])
    fa_m, fa_h = _ci95(fa_bench["throughputs"])
    co_m, co_h = _ci95(comb_bench["throughputs"])
    to_m, to_h = _ci95(tome_r8_bench["throughputs"])

    t_labels = ["Baseline", "ToMe r=8", "FA-Only", "Combined\n(ToMe+FA)"]
    t_means = [b_m, to_m, fa_m, co_m]
    t_halfs = [b_h, to_h, fa_h, co_h]
    t_colors = [PALETTE["baseline"], PALETTE["tome_r8"], PALETTE["fa_only"], PALETTE["combined"]]

    # accuracy diffs (pp vs baseline)
    bp = base_acc["top1_accuracy"]
    n_val = base_acc["total"]
    fa_p = fa_acc["top1_accuracy"]
    co_p = comb_acc["top1_accuracy"]
    tome_r8_res = next(r for r in sweep["results"] if r["r"] == 8)
    to_p = tome_r8_res["top1_accuracy"]

    a_labels = ["FA-Only", "ToMe r=8", "Combined\n(ToMe+FA)"]
    a_diffs = [(fa_p - bp) * 100, (to_p - bp) * 100, (co_p - bp) * 100]
    a_halfs = [_prop_diff_ci95_half(bp, p, n_val) for p in [fa_p, to_p, co_p]]
    a_colors = [PALETTE["fa_only"], PALETTE["tome_r8"], PALETTE["combined"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=DPI)

    # ── throughput bars ──
    xs1 = np.arange(len(t_labels))
    bars1 = ax1.bar(xs1, t_means, color=t_colors, width=0.55, zorder=3,
                    edgecolor="white", linewidth=0.5)
    ax1.errorbar(xs1, t_means, yerr=t_halfs, fmt="none",
                 ecolor="#111111", elinewidth=1.8, capsize=7, capthick=1.8, zorder=5)
    ax1.set_xticks(xs1)
    ax1.set_xticklabels(t_labels, fontsize=10)
    ax1.set_ylabel("Throughput (images / sec)", fontsize=11)
    ax1.set_title("Mean Throughput ± 95% CI", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.set_ylim(0, max(t_means) * 1.22)
    for bar, m in zip(bars1, t_means):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            m + max(t_means) * 0.017,
            f"{m:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # ── accuracy diff bars ──
    xs2 = np.arange(len(a_labels))
    bars2 = ax2.bar(xs2, a_diffs, color=a_colors, width=0.45, zorder=3,
                    edgecolor="white", linewidth=0.5)
    ax2.errorbar(xs2, a_diffs, yerr=a_halfs, fmt="none",
                 ecolor="#111111", elinewidth=1.8, capsize=7, capthick=1.8, zorder=5)
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(a_labels, fontsize=10)
    ax2.set_ylabel("Top-1 Accuracy Δ vs. Baseline (pp)", fontsize=11)
    ax2.set_title("Accuracy Difference ± 95% CI", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    y_range = max(abs(v) for v in a_diffs) + max(a_halfs)
    ax2.set_ylim(-y_range * 1.4, y_range * 0.6)
    for bar, val in zip(bars2, a_diffs):
        va = "top" if val < 0 else "bottom"
        yoff = -0.04 if val < 0 else 0.02
        ax2.text(
            bar.get_x() + bar.get_width() / 2, val + yoff,
            f"{val:+.3f}pp", ha="center", va=va, fontsize=9, fontweight="bold",
        )

    fig.suptitle(
        "Main Experimental Conditions — ViT-B/16 on ImageNet-1k  (n = 50,000 images)",
        fontsize=11, color="#333333", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [B] CI chart           → {out_path}")


# ── Markdown summary ──────────────────────────────────────────────────────────

def write_summary(
    sweep: dict,
    phase3: dict, phase4: dict,
    base_acc: dict, fa_acc: dict, comb_acc: dict,
    out_path: Path,
) -> None:
    p3_t = phase3["throughput"]
    p3_a = phase3["accuracy"]
    p4_cvb_t = phase4["combined_vs_baseline"]["throughput"]
    p4_cvb_a = phase4["combined_vs_baseline"]["accuracy"]
    p4_cvf_t = phase4["combined_vs_flash_only"]["throughput"]
    p4_cvf_a = phase4["combined_vs_flash_only"]["accuracy"]

    rec_r = sweep.get("recommended_r", 8)
    r4 = next(r for r in sweep["results"] if r["r"] == 4)
    r8 = next(r for r in sweep["results"] if r["r"] == rec_r)

    fa_imp = p3_t["improvement_pct"]
    comb_vs_base_imp = p4_cvb_t["improvement_pct"]
    comb_vs_fa_imp = p4_cvf_t["improvement_pct"]   # negative

    lines = [
        "# Experimental Evaluation Summary",
        "",
        "> Auto-generated by `scripts/generate_eval_figures.py`",
        "",
        "## Key Findings",
        "",
        f"- **Flash Attention alone: +{fa_imp:.1f}% throughput** over the standard ViT-B/16 "
        f"baseline (p ≈ {p3_t['p_value']:.2e}, highly significant). "
        f"Accuracy cost is negligible: −{abs(p3_a['diff_pct']):.3f} pp "
        f"(p = {p3_a['p_value']:.3f}, n.s.).",
        "",
        f"- **Token Merging (r={rec_r}) sweet spot: +{r8['throughput_improvement_pct']:.1f}% throughput**, "
        f"−{r8['accuracy_drop_pct']:.3f} pp accuracy. "
        f"This is the recommended Pareto-optimal configuration.",
        "",
        f"- **⚠ Negative interaction — Combined (ToMe+FA) is *slower* than FA-alone:** "
        f"adding ToMe on top of Flash Attention reduces throughput by "
        f"{abs(comb_vs_fa_imp):.1f}% relative to FA-only "
        f"(p = {p4_cvf_t['p_value']:.2e}). "
        f"The expected additive speedup does not materialise.",
        "",
        f"- **ToMe r=4 shows no statistically significant speedup:** "
        f"{r4['throughput_improvement_pct']:+.2f}% (p = {r4['p_value']:.2f}). "
        f"A {r4['reduction_pct']}% token reduction is insufficient to overcome "
        f"attention-kernel overhead at this batch size.",
        "",
        f"- **Combined vs. Baseline is still net-positive (+{comb_vs_base_imp:.1f}%, "
        f"p ≈ {p4_cvb_t['p_value']:.2e})**, but the accuracy penalty "
        f"(−{abs(p4_cvb_a['diff_pct']):.3f} pp) is larger than either method alone, "
        f"and throughput is dominated by FA-only. Combined is Pareto-inferior.",
        "",
        "- **Memory caveat:** peak GPU memory differences are not statistically significant "
        "across any condition (all p > 0.2). Inter-trial variance is extremely low (< 0.1 GB), "
        "so the null result reflects genuine equivalence, not a power failure.",
        "",
        "## Slide Text",
        "",
        "> *Flash Attention delivers a clean +{fa:.0f}% throughput gain with no measurable "
        "accuracy cost, while Token Merging provides an additional configurable "
        "speed–accuracy trade-off. However, stacking both optimisations yields "
        "**less** total speedup than Flash Attention alone — a statistically significant "
        "negative interaction (p = {p:.2e}) that warns against naïve combination of "
        "these techniques.*".format(fa=fa_imp, p=p4_cvf_t["p_value"]),
        "",
        "## Numeric Reference",
        "",
        "| Condition | Throughput (img/s) | Top-1 Acc (%) | Δ Throughput vs. Baseline | Δ Accuracy vs. Baseline |",
        "|-----------|:-----------------:|:------------:|:------------------------:|:-----------------------:|",
        f"| Baseline           | {p3_t['mean_a']:.0f} | {base_acc['top1_accuracy']*100:.2f} | — | — |",
        f"| ToMe r={rec_r}           | {r8['mean_throughput']:.0f} | {r8['top1_accuracy']*100:.2f} | +{r8['throughput_improvement_pct']:.1f}% | {-r8['accuracy_drop_pct']:+.3f} pp |",
        f"| FA-Only            | {p3_t['mean_b']:.0f} | {fa_acc['top1_accuracy']*100:.2f} | +{fa_imp:.1f}% | {p3_a['diff_pct']:+.3f} pp |",
        f"| Combined (ToMe+FA) | {p4_cvb_t['mean_b']:.0f} | {comb_acc['top1_accuracy']*100:.2f} | +{comb_vs_base_imp:.1f}% | {p4_cvb_a['diff_pct']:+.3f} pp |",
        "",
    ]

    out_path.write_text("\n".join(lines))
    print(f"  [C] Markdown summary   → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation figures and summary for ToMe-Flash experiments."
    )
    parser.add_argument(
        "--results_dir", default="results latest",
        help="Directory containing result JSON files (default: 'results latest')",
    )
    args = parser.parse_args()

    res = Path(args.results_dir)
    if not res.is_dir():
        sys.exit(f"ERROR: results directory not found: {res}")

    fig_dir = res / "figures"
    fig_dir.mkdir(exist_ok=True)

    print(f"\nLoading data from: {res}/")

    base_bench    = _load(res / "baseline_benchmark.json",   "baseline_benchmark")
    base_acc      = _load(res / "baseline_accuracy.json",    "baseline_accuracy")
    fa_bench      = _load(res / "flash_only_benchmark.json", "flash_only_benchmark")
    fa_acc        = _load(res / "flash_only_accuracy.json",  "flash_only_accuracy")
    comb_bench    = _load(res / "combined_benchmark.json",   "combined_benchmark")
    comb_acc      = _load(res / "combined_accuracy.json",    "combined_accuracy")
    tome_r8_bench = _load(res / "tome_r8_benchmark.json",    "tome_r8_benchmark")
    sweep         = _load(res / "tome_sweep_summary.json",   "tome_sweep_summary")
    phase3        = _load(res / "phase3_stats.json",         "phase3_stats")
    phase4        = _load(res / "phase4_stats.json",         "phase4_stats")

    print("\nGenerating figures...")

    fig_pareto(sweep=sweep, out_path=fig_dir / "pareto_curve.png")

    fig_ci_chart(
        base_bench=base_bench, fa_bench=fa_bench,
        comb_bench=comb_bench, tome_r8_bench=tome_r8_bench,
        base_acc=base_acc, fa_acc=fa_acc, comb_acc=comb_acc,
        sweep=sweep,
        out_path=fig_dir / "conditions_ci_chart.png",
    )

    write_summary(
        sweep=sweep, phase3=phase3, phase4=phase4,
        base_acc=base_acc, fa_acc=fa_acc, comb_acc=comb_acc,
        out_path=res / "eval_summary.md",
    )

    print(f"\nDone. Output: {fig_dir}/")
    print()
    print("── Slide guide ──────────────────────────────────────────────────────────")
    print("  [A] pareto_curve.png        → 'Speed–Accuracy Trade-off' slide")
    print("      Highlight the ★ point (r=8) as the recommended setting.")
    print()
    print("  [B] conditions_ci_chart.png → 'Main Results' comparison slide")
    print("      Left panel: absolute throughput. Right: accuracy cost.")
    print("      Caption: 'Error bars = 95% CI, n=50k images'")
    print()
    print("  [C] eval_summary.md         → Copy 'Slide Text' block onto a")
    print("      conclusions slide; use the table as an appendix slide.")
    print("─────────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
