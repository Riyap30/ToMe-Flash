#!/bin/bash
#SBATCH --job-name=vit_run_all
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/rsp9219/project/results/run_all_%j.out
#SBATCH --error=/scratch/rsp9219/project/results/run_all_%j.err

# -----------------------------------------------------------------------
# Full 2×2 factorial experiment (all four conditions, Phases 1–3)
# Submit with: sbatch slurm/run_all.slurm
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
        python run_all.py
    "

echo "---"
echo "End time    : $(date)"
echo "Exit status : $?"
