import argparse
import sys
from pathlib import Path

from .figure_builders import (
    fig_absolute_pareto,
    fig_ci_chart,
    fig_effect_sizes,
    fig_memory_nonsig,
    fig_pareto,
    fig_stat_forest,
    fig_violin_strip,
)
from .io_helpers import load_json
from .report_writers import write_ppt_tables, write_project_overview_table, write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation figures and summary for ToMe-Flash experiments.")
    parser.add_argument("--results_dir", default="final results", help="Directory containing result JSON files (default: 'final results')")
    args = parser.parse_args()

    res = Path(args.results_dir)
    if not res.is_dir():
        sys.exit(f"ERROR: results directory not found: {res}")
    fig_dir = res / "figures"
    fig_dir.mkdir(exist_ok=True)
    print(f"\nLoading data from: {res}/")

    base_bench = load_json(res / "baseline_benchmark.json", "baseline_benchmark")
    base_acc = load_json(res / "baseline_accuracy.json", "baseline_accuracy")
    fa_bench = load_json(res / "flash_only_benchmark.json", "flash_only_benchmark")
    fa_acc = load_json(res / "flash_only_accuracy.json", "flash_only_accuracy")
    comb_bench = load_json(res / "combined_benchmark.json", "combined_benchmark")
    comb_acc = load_json(res / "combined_accuracy.json", "combined_accuracy")
    tome_r8_bench = load_json(res / "tome_r8_benchmark.json", "tome_r8_benchmark")
    tome_r4_bench = load_json(res / "tome_r4_benchmark.json", "tome_r4_benchmark")
    tome_r12_bench = load_json(res / "tome_r12_benchmark.json", "tome_r12_benchmark")
    tome_r14_bench = load_json(res / "tome_r14_benchmark.json", "tome_r14_benchmark")
    tome_r16_bench = load_json(res / "tome_r16_benchmark.json", "tome_r16_benchmark")
    tome_r4_acc = load_json(res / "tome_r4_accuracy.json", "tome_r4_accuracy")
    tome_r8_acc = load_json(res / "tome_r8_accuracy.json", "tome_r8_accuracy")
    tome_r12_acc = load_json(res / "tome_r12_accuracy.json", "tome_r12_accuracy")
    tome_r14_acc = load_json(res / "tome_r14_accuracy.json", "tome_r14_accuracy")
    tome_r16_acc = load_json(res / "tome_r16_accuracy.json", "tome_r16_accuracy")
    sweep = load_json(res / "tome_sweep_summary.json", "tome_sweep_summary")
    phase3 = load_json(res / "phase3_stats.json", "phase3_stats")
    phase4 = load_json(res / "phase4_stats.json", "phase4_stats")

    print("\nGenerating figures...")
    fig_pareto(sweep=sweep, out_path=fig_dir / "pareto_curve.png")
    fig_ci_chart(base_bench=base_bench, fa_bench=fa_bench, comb_bench=comb_bench, tome_r8_bench=tome_r8_bench, base_acc=base_acc, fa_acc=fa_acc, comb_acc=comb_acc, sweep=sweep, out_path=fig_dir / "conditions_ci_chart.png")
    fig_violin_strip(base_bench=base_bench, fa_bench=fa_bench, comb_bench=comb_bench, tome_r8_bench=tome_r8_bench, phase3=phase3, phase4=phase4, sweep=sweep, out_path=fig_dir / "violin_strip.png")
    fig_absolute_pareto(base_bench=base_bench, fa_bench=fa_bench, comb_bench=comb_bench, tome_r4_bench=tome_r4_bench, tome_r8_bench=tome_r8_bench, tome_r12_bench=tome_r12_bench, tome_r14_bench=tome_r14_bench, tome_r16_bench=tome_r16_bench, base_acc=base_acc, fa_acc=fa_acc, comb_acc=comb_acc, tome_r4_acc=tome_r4_acc, tome_r8_acc=tome_r8_acc, tome_r12_acc=tome_r12_acc, tome_r14_acc=tome_r14_acc, tome_r16_acc=tome_r16_acc, out_path=fig_dir / "absolute_pareto.png")
    fig_effect_sizes(base_bench=base_bench, fa_bench=fa_bench, comb_bench=comb_bench, tome_r4_bench=tome_r4_bench, tome_r8_bench=tome_r8_bench, tome_r12_bench=tome_r12_bench, tome_r14_bench=tome_r14_bench, tome_r16_bench=tome_r16_bench, base_acc=base_acc, fa_acc=fa_acc, comb_acc=comb_acc, sweep=sweep, phase3=phase3, phase4=phase4, out_path=fig_dir / "effect_sizes.png")
    fig_stat_forest(phase3=phase3, phase4=phase4, sweep=sweep, out_path=fig_dir / "stat_forest_plot.png")
    fig_memory_nonsig(phase3=phase3, phase4=phase4, out_path=fig_dir / "memory_nonsig_chart.png")
    write_summary(sweep=sweep, phase3=phase3, phase4=phase4, base_acc=base_acc, fa_acc=fa_acc, comb_acc=comb_acc, out_path=res / "eval_summary.md")
    write_ppt_tables(sweep=sweep, phase3=phase3, phase4=phase4, out_dir=res / "tables")
    write_project_overview_table(base_bench=base_bench, fa_bench=fa_bench, comb_bench=comb_bench, sweep=sweep, base_acc=base_acc, fa_acc=fa_acc, comb_acc=comb_acc, out_dir=res / "tables")

    print(f"\nDone. Output: {fig_dir}/")
    print()
    print("── Slide guide ──────────────────────────────────────────────────────────")
    print("  [A] pareto_curve.png        → 'Speed–Accuracy Trade-off' slide (ToMe only)")
    print("  [B] conditions_ci_chart.png → 'Main Results' comparison slide")
    print("  [1] violin_strip.png        → 'Distribution / Did You Get Lucky?' slide")
    print("      Exposes bimodal baseline structure; validates Welch's t-test choice.")
    print()
    print("  [3] absolute_pareto.png     → 'Key Finding' slide")
    print("      FA-Only Pareto-dominates Combined — only Baseline, FA-Only, ToMe r=16")
    print("      lie on the true frontier. Combined is Pareto-inferior to FA-Only.")
    print()
    print("  [4] effect_sizes.png        → 'Practical Significance' slide")
    print("      Hedges' g (throughput) and Cohen's h (accuracy) for all comparisons.")
    print("      BH-corrected; hatching = not significant.")
    print()
    print("  [5] stat_forest_plot.png    → 'Statistical Significance' slide")
    print("      Forest-style CIs + p-values for the key claims in one view.")
    print()
    print("  [6] memory_nonsig_chart.png → 'Memory Stability' slide")
    print("      Peak-memory deltas with CIs crossing zero and non-significant p-values.")
    print()
    print("  [T] tables/stats_summary_for_ppt.{csv,md}")
    print("      Paste-ready statistical table for methods/appendix slides.")
    print()
    print("  [O] tables/project_overview_for_ppt.{csv,md}")
    print("      High-level results summary table for intro/conclusion slides.")
    print()
    print("  [C] eval_summary.md         → Conclusions slide text + appendix table")
    print("─────────────────────────────────────────────────────────────────────────")
