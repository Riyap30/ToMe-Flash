#!/bin/bash
#SBATCH --job-name=vit_combined
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/rsp9219/project/results/combined_%j.out
#SBATCH --error=/scratch/rsp9219/project/results/combined_%j.err

# -----------------------------------------------------------------------
# Phase 4: ToMe r=8 + FlashAttention-2 combined DeiT-B/16 benchmark
# Requires: baseline_benchmark.json, tome_r8 or tome_only results,
#           flash_only_benchmark.json to already exist in results/.
# Submit with: sbatch slurm/combined.slurm
# -----------------------------------------------------------------------

source /ext3/env.sh
conda activate /ext3/env

cd /scratch/rsp9219/project

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "Start time  : $(date)"
echo "---"

singularity exec \
    --nv \
    --overlay /scratch/rsp9219/project/overlay-25GB-500K.ext3:rw \
    --overlay /share/apps/datasets/imagenet/imagenet-val.sqf:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate /ext3/env
        cd /scratch/rsp9219/project
        python run_combined.py --batch_size 256 --num_workers 8 --n_warmup 100 --n_trials 100
    "

echo "---"
echo "End time    : $(date)"
echo "Exit status : $?"
