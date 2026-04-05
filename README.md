# Environment Setup on Greene HPC (Singularity + Conda)

This guide covers setting up a reproducible GPU environment on NYU HPC using a Singularity container with a persistent overlay filesystem. The stack targets FlashAttention-2, ToMe, and PyTorch on CUDA 12.1.

---

## Setup

### Get a Fresh Overlay (25GB)

Copy and decompress the base overlay image into your working directory:

```bash
cp /scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz .
gunzip overlay-25GB-500K.ext3.gz
```

### Start Your GPU Session

Request an interactive session on an A100 node before proceeding. FlashAttention-2 must be compiled on A100 hardware.

### Enter Singularity

```bash
singularity exec --nv --overlay overlay-25GB-500K.ext3:rw \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash
```

### Install Miniforge (Inside Singularity)

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
rm Miniforge3-Linux-x86_64.sh
```

### Create the Wrapper Script

This script ensures paths and cache directories are always set correctly inside the container:

```bash
cat <<EOT > /ext3/env.sh
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
export TMPDIR=/ext3/tmp
export PIP_CACHE_DIR=/ext3/pip_cache
EOT

mkdir -p /ext3/tmp /ext3/pip_cache
source /ext3/env.sh
```

### Create the Python 3.10 Environment

Python 3.10 is used for best compatibility with `tome` and `flash-attn`:

```bash
conda create -y -p /ext3/env python=3.10
source activate /ext3/env
```

### Install Libraries

> **Order matters.** PyTorch must be installed before FlashAttention-2.

```bash
# A. Foundations
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# B. Helper Tools
pip install timm wandb scipy matplotlib psutil

# C. ToMe (fixed version from GitHub)
pip install "git+https://github.com/facebookresearch/ToMe.git"

# D. FlashAttention-2 (takes ~10–15 mins)
pip install flash-attn --no-build-isolation
```

---

## Running Experiments

Slurm batch scripts are located in the `slurm/` directory. Submit jobs from the **project root**:

```bash
sbatch slurm/phase1.slurm
```

To monitor your job:

```bash
squeue -u $USER
```

To check output logs (assuming the slurm script writes to `logs/`):

```bash
tail -f logs/phase1.out
```

---

## Project Structure

```
.
├── slurm/
│   └── phase1.slurm       # Batch job for phase 1
├── overlay-25GB-500K.ext3 # Persistent container overlay (not committed)
└── README.md
```

---

## Notes

- Always `source /ext3/env.sh` at the start of any new Singularity session before activating the conda env.
- Re-entering the environment later requires only the `singularity exec` command followed by `source /ext3/env.sh` and `source activate /ext3/env`.
