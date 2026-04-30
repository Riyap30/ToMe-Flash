from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms

from .stats_helpers import (
    bh_correct,
    ci95,
    cohen_h,
    fmt_p,
    hedges_g,
    pareto_frontier,
    prop_ci95_half,
    prop_diff_ci95_half,
    stars,
)

PALETTE = {
    "baseline":  "#3A7FC1",
    "tome_r8":   "#E07B54",
    "fa_only":   "#4DAF7C",
    "combined":  "#C04A55",
    "highlight": "#F0A830",
}
_TOME_SHADES = ["#F5C982", "#E8A020", "#C87010", "#A05010", "#783010"]

DPI = 200

plt.rcParams.update(
    {
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "axes.titlepad":     12,
        "axes.labelpad":      6,
        "legend.fontsize":    9.5,
        "legend.framealpha":  0.92,
        "legend.edgecolor":  "#cccccc",
    }
)

_GRID = dict(linestyle="--", alpha=0.28, color="#c8c8c8")


def _save(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_pareto(sweep: dict, out_path: Path) -> None:
    results = sweep["results"]
    rec_r   = sweep.get("recommended_r", 8)
    xs = [r["accuracy_drop_pct"]          for r in results]
    ys = [r["throughput_improvement_pct"] for r in results]
    rs = [r["r"]                          for r in results]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=DPI)

    for xi, yi, r in zip(xs, ys, rs):
        is_rec = r == rec_r
        ax.scatter(
            xi, yi,
            c=PALETTE["highlight"] if is_rec else PALETTE["tome_r8"],
            s=260 if is_rec else 120,
            zorder=5, edgecolors="white", linewidths=1.2,
        )
        label = f"r={r}" + (" ★" if is_rec else "")
        dy    = 4.5 if yi < max(ys) - 5 else -7
        ax.annotate(
            label,
            xy=(xi, yi),
            xytext=(xi + 0.07, yi + dy),
            fontsize=10,
            fontweight="bold" if is_rec else "normal",
            color="#222222",
        )

    sorted_pts = sorted(zip(xs, ys))
    ax.plot(
        [p[0] for p in sorted_pts],
        [p[1] for p in sorted_pts],
        "--", color="#999999", linewidth=1.2, zorder=2, alpha=0.7,
    )
    ax.axhline(0, color="#666666", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axvline(0, color="#666666", linewidth=0.8, linestyle=":", alpha=0.6)

    ax.legend(
        handles=[
            mpatches.Patch(color=PALETTE["tome_r8"],  label="ToMe reduction levels"),
            mpatches.Patch(color=PALETTE["highlight"], label=f"Recommended r={rec_r}  ★"),
        ],
        loc="lower right",
    )
    ax.set_xlabel("Accuracy Drop vs. Baseline (pp)")
    ax.set_ylabel("Throughput Improvement vs. Baseline (%)")
    ax.set_title(
        "Token Merging: Throughput–Accuracy Pareto Frontier\n"
        "ViT-B/16 · ImageNet-1k · r ∈ {4, 8, 12, 14, 16}  (higher-right is better)"
    )
    ax.grid(**_GRID)
    fig.tight_layout()
    _save(fig, out_path)
    print(f"  [A] Pareto curve       → {out_path}")


def fig_ci_chart(
    base_bench: dict,
    fa_bench: dict,
    comb_bench: dict,
    tome_r8_bench: dict,
    base_acc: dict,
    fa_acc: dict,
    comb_acc: dict,
    sweep: dict,
    out_path: Path,
) -> None:
    b_m,  b_h  = ci95(base_bench["throughputs"])
    fa_m, fa_h = ci95(fa_bench["throughputs"])
    co_m, co_h = ci95(comb_bench["throughputs"])
    to_m, to_h = ci95(tome_r8_bench["throughputs"])

    t_labels = ["Baseline", "ToMe r=8", "FA-Only", "Combined\n(ToMe+FA)"]
    t_means  = [b_m,  to_m,  fa_m,  co_m]
    t_halfs  = [b_h,  to_h,  fa_h,  co_h]
    t_colors = [
        PALETTE["baseline"], PALETTE["tome_r8"],
        PALETTE["fa_only"],  PALETTE["combined"],
    ]

    bp    = base_acc["top1_accuracy"]
    n_val = base_acc["total"]
    fa_p  = fa_acc["top1_accuracy"]
    co_p  = comb_acc["top1_accuracy"]
    to_p  = next(r for r in sweep["results"] if r["r"] == 8)["top1_accuracy"]

    a_labels = ["FA-Only", "ToMe r=8", "Combined\n(ToMe+FA)"]
    a_diffs  = [(fa_p - bp) * 100, (to_p - bp) * 100, (co_p - bp) * 100]
    a_halfs  = [prop_diff_ci95_half(bp, p, n_val) for p in [fa_p, to_p, co_p]]
    a_colors = [PALETTE["fa_only"], PALETTE["tome_r8"], PALETTE["combined"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=DPI)

    # throughput panel
    xs1   = np.arange(len(t_labels))
    bars1 = ax1.bar(
        xs1, t_means, color=t_colors, width=0.55,
        zorder=3, edgecolor="white", linewidth=0.5,
    )
    ax1.errorbar(
        xs1, t_means, yerr=t_halfs, fmt="none",
        ecolor="#111111", elinewidth=1.8, capsize=7, capthick=1.8, zorder=5,
    )
    ax1.set_xticks(xs1)
    ax1.set_xticklabels(t_labels)
    ax1.set_ylabel("Throughput (images / sec)")
    ax1.set_title("Mean Throughput ± 95% CI")
    ax1.grid(axis="y", **_GRID)
    ax1.set_ylim(0, max(t_means) * 1.22)
    for bar, m in zip(bars1, t_means):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            m + max(t_means) * 0.017,
            f"{m:.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # accuracy panel
    xs2   = np.arange(len(a_labels))
    bars2 = ax2.bar(
        xs2, a_diffs, color=a_colors, width=0.45,
        zorder=3, edgecolor="white", linewidth=0.5,
    )
    ax2.errorbar(
        xs2, a_diffs, yerr=a_halfs, fmt="none",
        ecolor="#111111", elinewidth=1.8, capsize=7, capthick=1.8, zorder=5,
    )
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(a_labels)
    ax2.set_ylabel("Top-1 Accuracy Δ vs. Baseline (pp)")
    ax2.set_title("Accuracy Difference ± 95% CI")
    ax2.grid(axis="y", **_GRID)
    y_range = max(abs(v) for v in a_diffs) + max(a_halfs)
    ax2.set_ylim(-y_range * 1.4, y_range * 0.6)
    for bar, val in zip(bars2, a_diffs):
        va   = "top"   if val < 0 else "bottom"
        yoff = -0.04   if val < 0 else 0.02
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + yoff,
            f"{val:+.3f}pp",
            ha="center", va=va, fontsize=9, fontweight="bold",
        )

    fig.suptitle(
        "Main Experimental Conditions — ViT-B/16 on ImageNet-1k  (n = 50,000 images)",
        fontsize=11, color="#333333", y=1.01,
    )
    fig.tight_layout()
    _save(fig, out_path)
    print(f"  [B] CI chart           → {out_path}")


def fig_violin_strip(
    base_bench: dict,
    fa_bench: dict,
    comb_bench: dict,
    tome_r8_bench: dict,
    phase3: dict,
    phase4: dict,
    sweep: dict,
    out_path: Path,
) -> None:
    arrs = [
        np.asarray(base_bench["throughputs"],    float),
        np.asarray(tome_r8_bench["throughputs"], float),
        np.asarray(fa_bench["throughputs"],      float),
        np.asarray(comb_bench["throughputs"],    float),
    ]
    colors = [
        PALETTE["baseline"], PALETTE["tome_r8"],
        PALETTE["fa_only"],  PALETTE["combined"],
    ]
    gs = [
        0.0,
        hedges_g(arrs[1], arrs[0]),
        hedges_g(arrs[2], arrs[0]),
        hedges_g(arrs[3], arrs[0]),
    ]

    r8_res = next(r for r in sweep["results"] if r["r"] == 8)
    raw_ps = [
        r8_res["p_value"],
        phase3["throughput"]["p_value"],
        phase4["combined_vs_baseline"]["throughput"]["p_value"],
    ]
    p_adj  = bh_correct(raw_ps)

    xtick_labels = ["Baseline\n(reference)"]
    for lbl, g in zip(["ToMe r=8", "FA-Only", "Combined\n(ToMe+FA)"], gs[1:]):
        xtick_labels.append(f"{lbl}\ng = {g:+.2f}")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=DPI)

    parts = ax.violinplot(
        list(arrs), positions=list(range(4)),
        showmeans=False, showmedians=True, showextrema=False, widths=0.65,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.55)
        pc.set_edgecolor("#333333")
        pc.set_linewidth(0.8)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2.5)

    rng = np.random.default_rng(42)
    for i, arr in enumerate(arrs):
        jitter = rng.normal(0, 0.05, len(arr))
        ax.scatter(i + jitter, arr, alpha=0.28, s=8, c=colors[i], zorder=4, linewidths=0)

    all_max = max(a.max() for a in arrs)
    all_min = min(a.min() for a in arrs)
    span    = all_max - all_min
    tick_h  = span * 0.015
    bkt_ys  = [all_max + span * 0.06, all_max + span * 0.15, all_max + span * 0.24]

    for j, (cond_i, p_a, g) in enumerate(zip([1, 2, 3], p_adj, gs[1:])):
        y = bkt_ys[j]
        ax.plot(
            [0, 0, cond_i, cond_i],
            [y, y + tick_h, y + tick_h, y],
            lw=1.5, color="#222222", zorder=6,
        )
        ax.text(
            cond_i / 2, y + tick_h * 1.6,
            f"{stars(p_a)}   g = {g:+.2f}   p_adj = {p_a:.2e}",
            ha="center", va="bottom", fontsize=8.5, color="#111111", zorder=6,
        )

    ax.set_xticks(list(range(4)))
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel("Throughput (images / sec)")
    ax.set_title(
        "Per-Trial Throughput Distribution — All Main Conditions\n"
        "Violin = KDE of 100 trials; dots = individual trials; "
        "brackets = BH-corrected p-value + Hedges' g"
    )
    ax.grid(axis="y", **_GRID)
    ax.set_ylim(all_min * 0.97, all_max + span * 0.38)

    fig.tight_layout()
    _save(fig, out_path)
    print(f"  [1] Violin + strip     → {out_path}")


def fig_absolute_pareto(
    base_bench: dict,
    fa_bench: dict,
    comb_bench: dict,
    tome_r4_bench: dict,
    tome_r8_bench: dict,
    tome_r12_bench: dict,
    tome_r14_bench: dict,
    tome_r16_bench: dict,
    base_acc: dict,
    fa_acc: dict,
    comb_acc: dict,
    tome_r4_acc: dict,
    tome_r8_acc: dict,
    tome_r12_acc: dict,
    tome_r14_acc: dict,
    tome_r16_acc: dict,
    out_path: Path,
) -> None:
    conditions = [
        ("Baseline",           base_bench,     base_acc,     PALETTE["baseline"], "o", 12),
        ("FA-Only",            fa_bench,       fa_acc,       PALETTE["fa_only"],  "s", 12),
        ("Combined (ToMe+FA)", comb_bench,     comb_acc,     PALETTE["combined"], "D", 11),
        ("ToMe r=4",           tome_r4_bench,  tome_r4_acc,  _TOME_SHADES[0],    "^", 10),
        ("ToMe r=8",           tome_r8_bench,  tome_r8_acc,  _TOME_SHADES[1],    "^", 10),
        ("ToMe r=12",          tome_r12_bench, tome_r12_acc, _TOME_SHADES[2],    "^", 10),
        ("ToMe r=14",          tome_r14_bench, tome_r14_acc, _TOME_SHADES[3],    "^", 10),
        ("ToMe r=16",          tome_r16_bench, tome_r16_acc, _TOME_SHADES[4],    "^", 10),
    ]
    label_offsets = {
        "Baseline":            ( 0.03,   60),
        "FA-Only":             ( 0.03,   60),
        "Combined (ToMe+FA)":  ( 0.03, -160),
        "ToMe r=4":            (-0.28, -140),
        "ToMe r=8":            ( 0.03,   60),
        "ToMe r=12":           ( 0.03,   60),
        "ToMe r=14":           ( 0.03, -160),
        "ToMe r=16":           ( 0.03,   60),
    }

    fig, ax = plt.subplots(figsize=(11, 8), dpi=DPI)
    xs, ys = [], []

    for label, bench, acc, color, marker, ms in conditions:
        arr      = np.asarray(bench["throughputs"], float)
        t_mean, t_half = ci95(arr)
        acc_p    = acc.get("top1_accuracy") or acc.get("top1_proportion", 0.0)
        n_acc    = acc.get("total", 50000)
        acc_half = prop_ci95_half(acc_p, n_acc)

        x, y = acc_p * 100, t_mean
        xs.append(x)
        ys.append(y)

        ax.errorbar(
            x, y, xerr=acc_half, yerr=t_half, fmt="none",
            ecolor=color, elinewidth=1.6, capsize=5, capthick=1.5,
            alpha=0.85, zorder=4,
        )
        ax.scatter(
            x, y, c=color, marker=marker, s=ms**2,
            zorder=5, edgecolors="#111111", linewidths=0.9,
        )
        dx, dy = label_offsets.get(label, (0.03, 60))
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x + dx, y + dy),
            fontsize=8.5, color="#222222",
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.8),
        )

    pf_idx = pareto_frontier(xs, ys)
    pf_pts = sorted([(xs[i], ys[i]) for i in pf_idx])
    if len(pf_pts) >= 2:
        ax.plot(
            [p[0] for p in pf_pts],
            [p[1] for p in pf_pts],
            "--", color="#333333", lw=1.8, zorder=2, alpha=0.75,
        )
        ax.text(
            pf_pts[1][0] - 0.08,
            (pf_pts[0][1] + pf_pts[1][1]) / 2,
            "Pareto\nfrontier",
            ha="right", va="center", fontsize=8, color="#333333", style="italic",
        )

    ax.legend(
        handles=[
            mpatches.Patch(color=PALETTE["baseline"], label="Baseline"),
            mpatches.Patch(color=PALETTE["fa_only"],  label="FA-Only"),
            mpatches.Patch(color=PALETTE["combined"], label="Combined (ToMe+FA)"),
            mpatches.Patch(color=_TOME_SHADES[2],     label="ToMe sweep r ∈ {4,8,12,14,16}"),
            plt.Line2D([0], [0], ls="--", color="#333333", lw=1.8, label="Pareto frontier"),
        ],
        loc="lower left",
    )
    ax.set_xlabel("Top-1 Accuracy (%)")
    ax.set_ylabel("Throughput (images / sec)")
    ax.set_title(
        "Complete Pareto Frontier — All Conditions in Absolute Accuracy–Throughput Space\n"
        "ViT-B/16 · ImageNet-1k  (error bars = 95% CI; FA-Only Pareto-dominates Combined)"
    )
    ax.grid(**_GRID)
    ax.set_xlim(min(xs) - 0.4, max(xs) + 0.5)
    ax.set_ylim(min(ys) - 180, max(ys) + 280)

    fig.tight_layout()
    _save(fig, out_path)
    print(f"  [3] Absolute Pareto    → {out_path}")


def fig_effect_sizes(
    base_bench: dict,
    fa_bench: dict,
    comb_bench: dict,
    tome_r4_bench: dict,
    tome_r8_bench: dict,
    tome_r12_bench: dict,
    tome_r14_bench: dict,
    tome_r16_bench: dict,
    base_acc: dict,
    fa_acc: dict,
    comb_acc: dict,
    sweep: dict,
    phase3: dict,
    phase4: dict,
    out_path: Path,
) -> None:
    base_t = np.asarray(base_bench["throughputs"],     float)
    fa_t   = np.asarray(fa_bench["throughputs"],       float)
    comb_t = np.asarray(comb_bench["throughputs"],     float)
    r4_t   = np.asarray(tome_r4_bench["throughputs"],  float)
    r8_t   = np.asarray(tome_r8_bench["throughputs"],  float)
    r12_t  = np.asarray(tome_r12_bench["throughputs"], float)
    r14_t  = np.asarray(tome_r14_bench["throughputs"], float)
    r16_t  = np.asarray(tome_r16_bench["throughputs"], float)

    def _acc(d):
        return d.get("top1_accuracy") or d.get("top1_proportion", 0.0)

    base_p, fa_p, comb_p = _acc(base_acc), _acc(fa_acc), _acc(comb_acc)
    sv = {r["r"]: r for r in sweep["results"]}

    comp_labels = [
        "FA vs Base",
        "Comb vs Base",
        "Comb vs FA",
        "ToMe r=4 vs Base",
        "ToMe r=8 vs Base",
        "ToMe r=12 vs Base",
        "ToMe r=14 vs Base",
        "ToMe r=16 vs Base",
    ]
    thru_g = [
        hedges_g(fa_t,   base_t),
        hedges_g(comb_t, base_t),
        hedges_g(comb_t, fa_t),
        hedges_g(r4_t,   base_t),
        hedges_g(r8_t,   base_t),
        hedges_g(r12_t,  base_t),
        hedges_g(r14_t,  base_t),
        hedges_g(r16_t,  base_t),
    ]
    acc_h = [
        cohen_h(fa_p,                    base_p),
        cohen_h(comb_p,                  base_p),
        cohen_h(comb_p,                  fa_p),
        cohen_h(sv[4]["top1_accuracy"],  base_p),
        cohen_h(sv[8]["top1_accuracy"],  base_p),
        cohen_h(sv[12]["top1_accuracy"], base_p),
        cohen_h(sv[14]["top1_accuracy"], base_p),
        cohen_h(sv[16]["top1_accuracy"], base_p),
    ]
    thru_p_adj = bh_correct([
        phase3["throughput"]["p_value"],
        phase4["combined_vs_baseline"]["throughput"]["p_value"],
        phase4["combined_vs_flash_only"]["throughput"]["p_value"],
        sv[4]["p_value"],  sv[8]["p_value"],
        sv[12]["p_value"], sv[14]["p_value"], sv[16]["p_value"],
    ])
    acc_p_adj = bh_correct([
        phase3["accuracy"]["p_value"],
        phase4["combined_vs_baseline"]["accuracy"]["p_value"],
        phase4["combined_vs_flash_only"]["accuracy"]["p_value"],
        sv[4]["acc_p_value"],  sv[8]["acc_p_value"],
        sv[12]["acc_p_value"], sv[14]["acc_p_value"], sv[16]["acc_p_value"],
    ])
    alpha = 0.05

    def _draw_panel(ax, effect_sizes, p_adjs, title, xlabel):
        n           = len(effect_sizes)
        y_positions = list(range(n - 1, -1, -1))
        g_min, g_max = min(effect_sizes), max(effect_sizes)
        x_lo = min(g_min - abs(g_min) * 0.12, -0.3)
        x_hi = max(g_max + abs(g_max) * 0.12,  0.3)
        zone_a = 0.06

        ax.axvspan(x_lo, -0.8, alpha=zone_a,       color="#C44E52", zorder=0)
        ax.axvspan(-0.8, -0.5, alpha=zone_a,       color="#E8A020", zorder=0)
        ax.axvspan(-0.5, -0.2, alpha=zone_a * 0.6, color="#999999", zorder=0)
        ax.axvspan(-0.2,  0.2, alpha=zone_a * 0.4, color="#999999", zorder=0)
        ax.axvspan( 0.2,  0.5, alpha=zone_a * 0.6, color="#999999", zorder=0)
        ax.axvspan( 0.5,  0.8, alpha=zone_a,       color="#55A868", zorder=0)
        ax.axvspan( 0.8, x_hi, alpha=zone_a,       color="#4C72B0", zorder=0)

        for y, g, p in zip(y_positions, effect_sizes, p_adjs):
            significant = p < alpha
            color       = "#4C72B0" if g >= 0 else "#C44E52"
            ax.barh(
                y, g,
                color=color,
                alpha=0.80 if significant else 0.35,
                hatch=None if significant else "//",
                edgecolor="#333333", linewidth=0.8, height=0.55, zorder=3,
            )
            x_text = g + (0.02 * (x_hi - x_lo) / 2) * (1 if g >= 0 else -1)
            ax.text(
                x_text, y, stars(p),
                ha="left" if g >= 0 else "right",
                va="center", fontsize=8.2, color="#111111", zorder=5,
            )

        ax.axvline(0, color="black", linewidth=1.1, zorder=4)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(comp_labels, fontsize=8.2)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim(x_lo, x_hi)
        ax.grid(axis="x", **_GRID)

        y_top = n - 1 + 0.55
        for bnd, lbl in [
            (-0.8, "Large"), (-0.5, "Med."), (-0.2, "Small"),
            ( 0.0, "Negl."), ( 0.2, "Small"), ( 0.5, "Med."), (0.8, "Large"),
        ]:
            if x_lo < bnd < x_hi:
                ax.axvline(bnd, color="#cccccc", lw=0.6, ls=":", zorder=1)
                ax.text(
                    bnd, y_top, lbl,
                    ha="center", va="bottom", fontsize=6,
                    color="#777777", style="italic",
                )

        ax.legend(
            handles=[
                mpatches.Patch(color="#555555", alpha=0.8,
                               label="Significant (BH α=0.05)"),
                mpatches.Patch(facecolor="#555555", alpha=0.35, hatch="//",
                               edgecolor="#333333", label="Not significant"),
            ],
            fontsize=7.5, loc="lower right", borderpad=0.6,
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 7), dpi=DPI)
    _draw_panel(
        ax1, thru_g, thru_p_adj,
        "Throughput Effect Size (Hedges' g)\nvs reference condition",
        "Hedges' g  [positive = faster than reference]",
    )
    _draw_panel(
        ax2, acc_h, acc_p_adj,
        "Accuracy Effect Size (Cohen's h)\nvs reference condition",
        "Cohen's h  [negative = less accurate than reference]",
    )
    ax2.set_yticklabels([])
    fig.suptitle(
        "Practical vs Statistical Significance — Effect Sizes for All Pairwise Comparisons\n"
        "Zones: |effect| < 0.2 negligible | 0.2–0.5 small | 0.5–0.8 medium | ≥ 0.8 large  ·  "
        "Hatching = not significant after BH-FDR correction",
        fontsize=9.5, color="#333333", y=1.02,
    )
    fig.subplots_adjust(left=0.20, right=0.98, top=0.86, bottom=0.11, wspace=0.18)
    _save(fig, out_path)
    print(f"  [4] Effect sizes       → {out_path}")


def fig_stat_forest(phase3: dict, phase4: dict, sweep: dict, out_path: Path) -> None:
    rec_r = sweep.get("recommended_r", 8)
    r_rec = next(r for r in sweep["results"] if r["r"] == rec_r)

    def _tp_pct(stat):
        return (
            stat["ci_low"]  / stat["mean_a"] * 100,
            stat["ci_high"] / stat["mean_a"] * 100,
        )

    tp3_lo,  tp3_hi  = _tp_pct(phase3["throughput"])
    tp4b_lo, tp4b_hi = _tp_pct(phase4["combined_vs_baseline"]["throughput"])
    tp4f_lo, tp4f_hi = _tp_pct(phase4["combined_vs_flash_only"]["throughput"])

    rows = [
        {
            "label":   "TP: FA vs Base",
            "effect":  phase3["throughput"]["improvement_pct"],
            "ci_low":  tp3_lo,  "ci_high": tp3_hi,
            "p":       phase3["throughput"]["p_value"],
        },
        {
            "label":   f"TP: ToMe r={rec_r} vs Base",
            "effect":  r_rec["throughput_improvement_pct"],
            "ci_low":  np.nan,  "ci_high": np.nan,
            "p":       r_rec["p_value_adj"],
        },
        {
            "label":   "TP: Comb vs Base",
            "effect":  phase4["combined_vs_baseline"]["throughput"]["improvement_pct"],
            "ci_low":  tp4b_lo, "ci_high": tp4b_hi,
            "p":       phase4["combined_vs_baseline"]["throughput"]["p_value"],
        },
        {
            "label":   "TP: Comb vs FA",
            "effect":  phase4["combined_vs_flash_only"]["throughput"]["improvement_pct"],
            "ci_low":  tp4f_lo, "ci_high": tp4f_hi,
            "p":       phase4["combined_vs_flash_only"]["throughput"]["p_value"],
        },
        {
            "label":   "Acc: FA vs Base",
            "effect":  phase3["accuracy"]["diff_pct"],
            "ci_low":  phase3["accuracy"]["ci_low"],
            "ci_high": phase3["accuracy"]["ci_high"],
            "p":       phase3["accuracy"]["p_value"],
        },
        {
            "label":   f"Acc: ToMe r={rec_r} vs Base",
            "effect":  -r_rec["accuracy_drop_pct"],
            "ci_low":  np.nan,  "ci_high": np.nan,
            "p":       r_rec["acc_p_value_adj"],
        },
        {
            "label":   "Acc: Comb vs Base",
            "effect":  phase4["combined_vs_baseline"]["accuracy"]["diff_pct"],
            "ci_low":  phase4["combined_vs_baseline"]["accuracy"]["ci_low"],
            "ci_high": phase4["combined_vs_baseline"]["accuracy"]["ci_high"],
            "p":       phase4["combined_vs_baseline"]["accuracy"]["p_value"],
        },
        {
            "label":   "Acc: Comb vs FA",
            "effect":  phase4["combined_vs_flash_only"]["accuracy"]["diff_pct"],
            "ci_low":  phase4["combined_vs_flash_only"]["accuracy"]["ci_low"],
            "ci_high": phase4["combined_vs_flash_only"]["accuracy"]["ci_high"],
            "p":       phase4["combined_vs_flash_only"]["accuracy"]["p_value"],
        },
    ]

    fig, ax = plt.subplots(figsize=(13.8, 8.0), dpi=DPI)
    y_vals = np.arange(len(rows))[::-1]
    effects = np.asarray([r["effect"] for r in rows], float)
    has_ci = [np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]) for r in rows]
    ci_lows = np.asarray([r["ci_low"] for r in rows if np.isfinite(r["ci_low"])], float)
    ci_highs = np.asarray([r["ci_high"] for r in rows if np.isfinite(r["ci_high"])], float)

    if len(ci_lows) > 0 and len(ci_highs) > 0:
        x_min = min(np.min(effects), np.min(ci_lows))
        x_max = max(np.max(effects), np.max(ci_highs))
    else:
        x_min = np.min(effects)
        x_max = np.max(effects)
    span = max(x_max - x_min, 1.0)
    x_pad = span * 0.18
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    # Visual grouping to separate throughput and accuracy claims at a glance.
    ax.axhspan(3.5, 7.5, color="#EAF2FB", alpha=0.45, zorder=0)
    ax.axhspan(-0.5, 3.5, color="#F7F4EA", alpha=0.45, zorder=0)
    ax.axhline(3.5, color="#bbbbbb", lw=1.0, ls=":", zorder=1)

    for y, row, draw_ci in zip(y_vals, rows, has_ci):
        effect = row["effect"]
        p = row["p"]
        significant = p < 0.05
        color = "#1f77b4" if significant else "#8f8f8f"
        marker = "o" if significant else "s"

        if draw_ci:
            lo_err = abs(effect - row["ci_low"])
            hi_err = abs(row["ci_high"] - effect)
            ax.errorbar(
                effect, y,
                xerr=np.array([[lo_err], [hi_err]]),
                fmt=marker, color=color, ecolor=color,
                markersize=6.5, elinewidth=1.8, capsize=4, zorder=4,
            )
        else:
            ax.scatter(effect, y, s=46, c=color, marker=marker, zorder=5)

        # Place p-value labels near each point (instead of one crowded right column).
        text_dx = span * 0.06
        label_x = effect + (text_dx if effect >= 0 else -text_dx)
        ha = "left" if effect >= 0 else "right"
        ax.text(
            label_x, y,
            f"p={fmt_p(p)} {stars(p)}",
            va="center", ha=ha, fontsize=8.5, color="#222222", zorder=6,
        )

    ax.axvline(0, color="#111111", lw=1.0, ls="--", alpha=0.7, zorder=2)
    ax.set_yticks(y_vals)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Effect (Throughput: % change | Accuracy: pp change)")
    ax.set_title(
        "Statistical Test Forest Plot — Main Claims at a Glance\n"
        "Error bars = 95% CI where available; p-values are shown next to each estimate"
    )
    ax.grid(axis="x", **_GRID)

    tf = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.01, 7.35, "Throughput claims", transform=tf, ha="left", va="center", fontsize=8.8, color="#365f8c")
    ax.text(0.01, 3.35, "Accuracy claims", transform=tf, ha="left", va="center", fontsize=8.8, color="#7d692a")

    fig.subplots_adjust(left=0.28, right=0.97, top=0.90, bottom=0.11)
    _save(fig, out_path)
    print(f"  [5] Stat forest        → {out_path}")


def fig_memory_nonsig(phase3: dict, phase4: dict, out_path: Path) -> None:
    def _mem(stat):
        return {
            "delta_pct":   stat["improvement_pct"],
            "ci_low_pct":  stat["ci_low"]  / stat["mean_a"] * 100,
            "ci_high_pct": stat["ci_high"] / stat["mean_a"] * 100,
        }

    rows = [
        {"label": "FA vs Base",       "p": phase3["memory"]["p_value"],
         **_mem(phase3["memory"])},
        {"label": "Comb vs Base",     "p": phase4["combined_vs_baseline"]["memory"]["p_value"],
         **_mem(phase4["combined_vs_baseline"]["memory"])},
        {"label": "Comb vs ToMe r=8", "p": phase4["combined_vs_tome_r8"]["memory"]["p_value"],
         **_mem(phase4["combined_vs_tome_r8"]["memory"])},
        {"label": "Comb vs FA",       "p": phase4["combined_vs_flash_only"]["memory"]["p_value"],
         **_mem(phase4["combined_vs_flash_only"]["memory"])},
    ]

    vals      = [r["delta_pct"]               for r in rows]
    yerr_low  = [abs(v - r["ci_low_pct"])  for v, r in zip(vals, rows)]
    yerr_high = [abs(r["ci_high_pct"] - v) for v, r in zip(vals, rows)]
    colors    = ["#8DA0CB" if r["p"] >= 0.05 else "#FC8D62" for r in rows]

    fig, ax = plt.subplots(figsize=(9.8, 5.8), dpi=DPI)
    xs   = np.arange(len(rows))
    bars = ax.bar(
        xs, vals, color=colors, width=0.55,
        edgecolor="#222222", linewidth=0.7, zorder=3,
    )
    ax.errorbar(
        xs, vals,
        yerr=np.array([yerr_low, yerr_high]),
        fmt="none", ecolor="#222222", elinewidth=1.5,
        capsize=5, capthick=1.5, zorder=4,
    )
    ax.axhline(0, color="#111111", lw=1.0, ls="--", alpha=0.8)

    for i, (bar, row) in enumerate(zip(bars, rows)):
        ypos = vals[i] + (0.12 if vals[i] >= 0 else -0.12)
        va   = "bottom" if vals[i] >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2, ypos,
            f"p={fmt_p(row['p'])}",
            ha="center", va=va, fontsize=8.5, color="#222222",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows])
    ax.set_ylabel("Peak Memory Delta (%)")
    ax.set_title(
        "Peak Memory Differences Are Not Statistically Significant\n"
        "All 95% CIs cross 0 and all p-values are non-significant"
    )
    ax.grid(axis="y", **_GRID)
    ax.legend(
        handles=[
            mpatches.Patch(color="#8DA0CB", label="Not significant (p ≥ 0.05)"),
            mpatches.Patch(color="#FC8D62", label="Significant (p < 0.05)"),
        ],
        loc="upper right",
    )

    fig.tight_layout()
    _save(fig, out_path)
    print(f"  [6] Memory chart       → {out_path}")
