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
pip install flash-attn==2.4.2 --no-build-isolation
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

### 5.2 Statistical highlights
- **FA-Only vs Baseline (throughput):** +59.1%, p ~ 7.33e-79 (highly significant).
- **FA-Only vs Baseline (accuracy):** -0.016 pp, p = 0.9478 (not significant).
- **ToMe r=8 vs Baseline (throughput):** +18.1%, p ~ 8.47e-61 (significant).
- **Combined vs FA-Only (throughput):** -9.05%, p ~ 1.61e-14 (significant slowdown).
- **Memory differences:** no significant changes across key comparisons (all p > 0.19 in the main pairwise analyses).

Source files:
- `final results/phase3_stats.json`
- `final results/phase4_stats.json`
- `final results/tables/stats_summary_for_ppt.md`

### 5.3 ToMe sweep behavior (r = 4, 8, 12, 14, 16)
- Throughput rises with larger `r`, but accuracy drops progressively.
- `r=8` is selected as the recommended compromise in this project.
- `r=16` is fastest but produces a substantial accuracy loss.

Source file:
- `final results/tome_sweep_summary.json`

### 5.4 Figures

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

Quick links:
- [Pareto curve](final%20results/figures/pareto_curve.png)
- [Effect sizes](final%20results/figures/effect_sizes.png)
- [Violin/strip distribution plot](final%20results/figures/violin_strip.png)

### 5.5 Observations

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
