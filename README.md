# Efficient Vision Transformer Inference via Token Merging and FlashAttention

This repository implements and evaluates an inference-time optimization framework for Vision Transformers (ViTs), centered on **Token Merging (ToMe)** and **FlashAttention-2**.

The project studies whether combining:
- token-count reduction (ToMe), and
- IO-aware exact attention kernels (FlashAttention-2),

produces additive gains in throughput and memory efficiency while preserving ImageNet-1K accuracy.

The core experimental model is **DeiT-B/16** evaluated on **ImageNet-1K validation (50,000 images)**, with a **2x2 factorial design**:

| Condition | No FlashAttention | FlashAttention-2 |
|---|---|---|
| No ToMe | Baseline | FA-Only |
| ToMe | ToMe-Only | Combined (ToMe+FA) |

---

## 1) Project Description

### Objective
Vision Transformers scale quadratically with token sequence length, which limits inference efficiency. This project evaluates two orthogonal optimizations:

- **ToMe**: progressively merges redundant tokens in each transformer block (reduces compute).
- **FlashAttention-2**: replaces standard attention with an IO-aware kernel (reduces memory traffic bottlenecks).

### What this repository delivers
- Reproducible baseline and optimized inference runs on A100 GPU.
- Independent and joint evaluation of ToMe and FlashAttention.
- Statistical testing for throughput, memory, and accuracy differences.
- Figure/table generation for reporting and presentation.

### Experimental setup (implemented)
- **Hardware**: NVIDIA A100 (Ampere; required for FlashAttention-2 support in this setup).
- **Software**: PyTorch 2.x + CUDA 12.1, `timm`, ToMe, flash-attn.
- **Dataset**: ImageNet-1K validation split.
- **Metrics**:
  - Throughput (images/sec)
  - Peak GPU memory (GB)
  - Top-1 (and Top-5) accuracy
- **Statistics**:
  - Welch's t-test for throughput/memory
  - Two-proportion z-test for accuracy
  - alpha = 0.05

---

## 2) Milestones and Completion Status

Milestones are based on the proposal and mapped to implemented project phases.

| Milestone | Description | Status | Evidence |
|---|---|---|---|
| M1: Reproducible baseline pipeline | Baseline DeiT-B/16 benchmark + accuracy on ImageNet-1K val | Completed | `run_phase1.py`, `final results/baseline_benchmark.json`, `final results/baseline_accuracy.json` |
| M2: ToMe-only integration and sweep | ToMe patching and sweep across merge ratios | Completed | `run_token_merging.py`, `final results/tome_sweep_summary.json` |
| M3: FlashAttention-only integration | FA-2 model path with benchmark + accuracy + stats vs baseline | Completed | `run_flash_attention.py`, `final results/phase3_stats.json` |
| M4: Joint ToMe+FA evaluation | Combined condition + pairwise stats vs all prior conditions | Completed | `run_combined.py`, `final results/phase4_stats.json` |
| M5: Statistical significance analysis | Hypothesis tests and confidence intervals across conditions | Completed | `stats.py`, `final results/tables/stats_summary_for_ppt.md`, `final results/eval_summary.md` |
| M6: Final visualization/reporting | Publication/presentation-ready figures and summary tables | Completed | `scripts/generate_eval_figures.py`, `final results/figures/`, `final results/tables/` |
| M7: Live inference demo (optional) | Real-time side-by-side app demo | Not implemented in this repo | No deployed demo app code in current tree |

---

## 3) Repository and Code Structure

### Core experiment runners
- `run_phase1.py` - baseline run (benchmark + accuracy).
- `run_token_merging.py` - ToMe sweep (`r` values; includes r=0 sanity check support).
- `run_flash_attention.py` - FlashAttention-only run.
- `run_combined.py` - combined ToMe + FlashAttention run.
- `run_profiler.py` - profiling utilities/traces.

### Supporting modules
- `models/`
  - `baseline.py` - unmodified DeiT path.
  - `tome_only.py` - ToMe-patched model loading.
  - `flash_only.py` - FA-only model loading.
  - `combined.py` - joint ToMe + FlashAttention model loading.
- `data/imagenet_loader.py` - ImageNet val loader + preprocessing and fast-loader mode.
- `benchmark.py` - throughput and memory benchmarking logic.
- `evaluate.py` - top-1/top-5 accuracy evaluation.
- `stats.py` - statistical tests and effect-size helpers.
- `meta.py` - run metadata capture.

### Job orchestration and automation
- `slurm/`
  - `phase1.slurm`
  - `token_merging.slurm`
  - `flash_attention.slurm`
  - `combined.slurm`
  - `run_all.slurm`
  - `profiler.slurm`

### Analysis and figure generation
- `scripts/generate_eval_figures.py` - generates report figures/tables.
- `scripts/eval_figures/` - plotting/statistics/report writer pipeline.

### Results directories
- `final results/` - finalized result artifacts used in report/presentation.
- `results/`, `results latest/`, `results 256/` - run outputs/intermediate exports.
- `profiles/` - profiler traces.

---

## 4) Setup and Example Commands

### Environment setup (NYU HPC / Singularity workflow)

> This project was executed in a Singularity + overlay workflow on A100 nodes.

### 4.1 Prepare overlay
```bash
cp /scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz .
gunzip overlay-25GB-500K.ext3.gz
```

### 4.2 Enter Singularity
```bash
singularity exec --nv --overlay overlay-25GB-500K.ext3:rw \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash
```

### 4.3 Install Miniforge and create env
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
rm Miniforge3-Linux-x86_64.sh

cat <<'EOT' > /ext3/env.sh
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
export TMPDIR=/ext3/tmp
export PIP_CACHE_DIR=/ext3/pip_cache
EOT

mkdir -p /ext3/tmp /ext3/pip_cache
source /ext3/env.sh
conda create -y -p /ext3/env python=3.10
conda activate /ext3/env
```

### 4.4 Install dependencies
```bash
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/ToMe.git"

# D. FlashAttention-2 (takes ~10–15 mins)
pip install flash-attn --no-build-isolation
```

---

## Configuration

Before submitting jobs, verify these settings in each `slurm/*.slurm` file:

| Setting | Where | What to change |
|---|---|---|
| Account | `#SBATCH --account` | Your NYU HPC allocation (e.g. `ece_gy_9143-2026sp`) |
| Project directory | Shell variable `PROJECT_NAME` | Name of your folder under `/scratch/$USER/` (default: `ToMe-Flash`) |
| NetID in output paths | `#SBATCH --output/--error` | Uses `%u` — resolved automatically by SLURM, no change needed |

To override the project directory without editing the scripts:

```bash
PROJECT_NAME=my_custom_dir sbatch slurm/phase1.slurm
```

### Run commands (from repo root)

### Baseline (Phase 1)
```bash
python run_phase1.py --batch_size 256 --num_workers 8 --device cuda --n_warmup 100 --n_trials 100
```

### ToMe sweep (Phase 2)
```bash
python run_token_merging.py --r_values 4 8 12 14 16 --batch_size 256 --num_workers 8 --device cuda --n_warmup 100 --n_trials 100
```

### FlashAttention-only (Phase 3)
```bash
python run_flash_attention.py --batch_size 256 --num_workers 8 --device cuda --n_warmup 100 --n_trials 100
```

### Combined ToMe + FlashAttention (Phase 4)
```bash
python run_combined.py --r 8 --batch_size 256 --num_workers 8 --device cuda --n_warmup 100 --n_trials 100
```

### Generate evaluation figures/tables
```bash
python scripts/generate_eval_figures.py --results_dir "final results"
```

### Smoke-test mode
All major scripts support `--fast` to run a short debug pass:
```bash
python run_combined.py --fast
```

### Slurm examples
```bash
sbatch slurm/phase1.slurm
sbatch slurm/token_merging.slurm
sbatch slurm/flash_attention.slurm
sbatch slurm/combined.slurm
```

---

## 5) Results and Observations

All finalized artifacts are in `final results/`.

### 5.1 Main quantitative results

| Condition | Throughput (img/s) | Top-1 Acc (%) | Delta Throughput vs Baseline | Delta Accuracy vs Baseline |
|---|---:|---:|---:|---:|
| Baseline | 2545.4 | 81.75 | - | - |
| ToMe r=8 | 3005.7 | 81.15 | +18.1% | -0.602 pp |
| FA-Only | 4048.6 | 81.73 | +59.1% | -0.016 pp |
| Combined (ToMe+FA) | 3682.0 | 80.62 | +44.7% | -1.134 pp |

Source files:
- `final results/tables/project_overview_for_ppt.md`
- `final results/eval_summary.md`

### 5.2 Pairwise statistical results (alpha = 0.05)

| Comparison | Metric | Effect | 95% CI | p-value | Significant |
|---|---|---:|---:|---:|---:|
| FA-Only vs Baseline | Throughput | +59.052% | [+56.209, +61.896]% | 7.33e-79 | Yes |
| ToMe r=8 vs Baseline | Throughput | +18.080% | N/A (not serialized) | 1.06e-60 | Yes |
| Combined vs Baseline | Throughput | +44.652% | [+42.137, +47.167]% | 7.87e-74 | Yes |
| Combined vs FA-Only | Throughput | -9.054% | [-11.202, -6.906]% | 1.61e-14 | Yes |
| FA-Only vs Baseline | Accuracy | -0.016 pp | [-0.495, +0.463] pp | 0.9478 | No |
| ToMe r=8 vs Baseline | Accuracy | -0.602 pp | N/A (not serialized) | 0.0179 | Yes |
| Combined vs Baseline | Accuracy | -1.134 pp | [-1.618, -0.650] pp | 4.49e-06 | Yes |
| Combined vs FA-Only | Accuracy | -1.118 pp | [-1.603, -0.633] pp | 6.13e-06 | Yes |

### 5.3 Memory and Top-5 summary

| Condition | Mean Peak Memory (GB) | Top-5 Accuracy (%) |
|---|---:|---:|
| Baseline | 1.2060 | 95.568 |
| ToMe r=8 | 1.2158 | 95.160 |
| FA-Only | 1.1943 | 95.568 |
| Combined (ToMe+FA) | 1.2029 | 94.972 |

Interpretation:
- Peak memory differences are small and statistically non-significant in main pairwise tests.
- Top-5 accuracy remains high (>94.9%) across all conditions, but the combined setup shows the largest drop.

Source files:
- `final results/phase3_stats.json`
- `final results/phase4_stats.json`
- `final results/tables/stats_summary_for_ppt.md`

### 5.4 ToMe sweep behavior (r = 4, 8, 12, 14, 16)

| r | Final tokens | Token reduction | Throughput (img/s) | Delta throughput vs baseline | Top-1 Accuracy (%) | Delta top-1 vs baseline |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 149 | 24.4% | 2473.1 | -2.84% | 81.588 | -0.162 pp |
| 8 | 101 | 48.7% | 3005.7 | +18.08% | 81.148 | -0.602 pp |
| 12 | 53 | 73.1% | 3571.2 | +40.30% | 80.166 | -1.584 pp |
| 14 | 29 | 85.3% | 3916.9 | +53.88% | 79.294 | -2.456 pp |
| 16 | 5 | 97.5% | 4197.5 | +64.90% | 77.580 | -4.170 pp |

Interpretation:
- Throughput improves monotonically as `r` increases.
- Accuracy degrades progressively with aggressive token merging.
- `r=8` is chosen as the project-recommended operating point because it delivers a meaningful speedup with moderate accuracy loss.

Source file:
- `final results/tome_sweep_summary.json`

### 5.5 Figures

The following figures are available in `final results/figures/`:

- Pareto and distribution views:
  - `final results/figures/pareto_curve.png`
  - `final results/figures/absolute_pareto.png`
  - `final results/figures/violin_strip.png`
- Statistical effect views:
  - `final results/figures/effect_sizes.png`
  - `final results/figures/stat_forest_plot.png`
  - `final results/figures/conditions_ci_chart.png`
- Memory-specific summary:
  - `final results/figures/memory_nonsig_chart.png`

#### Inline charts

![Pareto curve](final%20results/figures/pareto_curve.png)

![Effect sizes](final%20results/figures/effect_sizes.png)

![Violin/strip distribution plot](final%20results/figures/violin_strip.png)

### 5.6 Observations

1. **FlashAttention-2 is the strongest single optimization** in this setup, giving the best speedup while preserving accuracy.
2. **ToMe provides a tunable speed-accuracy trade-off**; moderate merging (`r=8`) is practical, but aggressive merging harms accuracy.
3. **The combined method is not additive here**: although faster than baseline, it is significantly slower than FA-only and less accurate.
4. **Peak memory was largely unchanged** across conditions under this measurement protocol.
5. **Practical recommendation from current results**: use FA-only for highest throughput with near-baseline accuracy; use ToMe only when an explicit speed-accuracy trade-off is desired.

---

## Notes and Reproducibility Caveats

- Several scripts assume HPC paths (for example `/scratch/rsp9219/project` and `/imagenet/val`). If running elsewhere, update path constants and/or job scripts.
- FlashAttention-2 build and runtime require compatible CUDA/PyTorch/GPU combinations; this repo was validated on A100-class hardware.
- ToMe + FA interaction can depend on sequence length patterns after merging; observed non-additivity is a key empirical result of this project.
