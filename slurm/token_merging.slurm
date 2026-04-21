#!/bin/bash
#SBATCH --job-name=vit_token_merging
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/rsp9219/project/results/token_merging_%j.out
#SBATCH --error=/scratch/rsp9219/project/results/token_merging_%j.err

# -----------------------------------------------------------------------
# Token Merging sweep: r ∈ {4, 8, 16} on DeiT-B/16, ImageNet-1K val
# Submit with: sbatch slurm/token_merging.slurm
# Prerequisite: run_phase1.py must have run and baseline JSONs must exist
# -----------------------------------------------------------------------

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
    /bin/bash << 'SINGEOF'
set -e
source /ext3/env.sh
conda activate /ext3/env
cd /scratch/rsp9219/project
echo "[slurm] Python: $(which python)"
echo "[slurm] tome check: $(python -c 'import tome; print(tome.__file__)' 2>&1)"
python -u run_token_merging.py \
    --r_values 4 8 12 14 16 \
    --batch_size 256 \
    --num_workers 8 \
    --n_warmup 100 \
    --n_trials 100
SINGEOF

SING_EXIT=$?
echo "---"
echo "End time    : $(date)"
echo "Exit status : $SING_EXIT"
exit $SING_EXIT
