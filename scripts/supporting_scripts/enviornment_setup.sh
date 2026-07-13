#!/bin/bash
#SBATCH --job-name=environment_setup
#SBATCH --output=out_environment_setup.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G

set -euo pipefail

# Make conda available in non-interactive shells (Slurm)
source "$(conda info --base)/etc/profile.d/conda.sh"

# Optional but recommended on HPC to avoid module/lib contamination
# module purge || true

ENV_NAME="med_vlm_mia_venv"

# Remove env if it exists (don't error if it doesn't)
conda env remove -n "${ENV_NAME}" -y || true

# Create env
conda create -n "${ENV_NAME}" -y python=3.10 pip
conda activate "${ENV_NAME}"

# Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

# --- IMPORTANT: compiled scientific stack via conda-forge (consistent ABI) ---
# Put ALL of these on conda-forge to avoid numpy/scipy/sklearn mismatch
conda install -y -c conda-forge \
  numpy=1.24.* \
  scipy \
  scikit-learn=1.2.* \
  scikit-image \
  pyarrow \
  pandas


python -m pip install python -m pip install -U hydra-core omegaconf


python -m pip install pyparsing

python -m pip install iopath

python -m pip install timm==1.0.3

python -m pip install opencv-python==4.6.0.66

python -m pip install webdataset

python -m pip install visual-genome

python -m pip install transformers==4.51.2

python -m pip install peft==0.4.0

python -m pip install decord==0.6.0

python -m pip install wandb

python -m pip install datasets==2.21.0

pip install nibabel

pip install ffmpeg-python

# (Optional) prefer conda OpenCV to avoid manylinux quirks
# conda install -y -c conda-forge opencv

# --- PyTorch wheels (CUDA 11.8) via pip ---
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.4.0 torchvision==0.19.0

# Flash-attn (builds against torch); keep after torch is installed
python -m pip install flash-attn==2.7.3 --no-build-isolation --upgrade

# Core ML tooling
python -m pip install transformers==4.51.2 accelerate==1.7.0 datasets==2.21.0 tokenizers>=0.21,<0.22

# Video / imaging (keep these pip unless you chose conda opencv above)
python -m pip install decord==0.6.0 ffmpeg-python imageio==2.34.0 imageio-ffmpeg==0.4.9 moviepy==1.0.3
python -m pip install opencv-python==4.6.0.66

# Medical imaging
python -m pip install nibabel==5.3.2

# Hydra / config
python -m pip install -U hydra-core omegaconf

# Misc deps
python -m pip install iopath webdataset visual-genome

# Install your project requirements LAST, but prevent it from re-installing numpy/scipy/sklearn
# Best: remove those pins from requirements.txt (recommended below)
python -m pip install -r /home/clo37/priv/MED-VLM-MIA/requirements.txt --no-deps

# Then install the remaining deps normally (will fill in what's missing without clobbering conda numpy/scipy)
python -m pip install -r /home/clo37/priv/MED-VLM-MIA/requirements.txt

# Sanity check: this should succeed
python -c "import numpy, scipy, sklearn; import scipy.special; print('OK', numpy.__version__, scipy.__version__, sklearn.__version__)"