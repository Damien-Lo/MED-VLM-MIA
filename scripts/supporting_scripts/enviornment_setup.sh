#!/bin/bash
#SBATCH --job-name=environment_setup
#SBATCH --output=out_environment_setup.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G

conda remove --name med_vlm_mia_venv

conda create -n med_vlm_mia_venv python=3.10 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate med_vlm_mia_venv



# PyTorch and torchvision for CUDA 11.8
pip install torch==2.4.0 torchvision==0.19.0 --extra-index-url https://download.pytorch.org/whl/cu118

# Flash-attn pinned to a compatible version
pip install flash-attn==2.7.3 --no-build-isolation --upgrade

# Transformers and accelerate
pip install transformers==4.51.2 accelerate==1.7.0

# Video processing dependencies
pip install decord ffmpeg-python imageio opencv-python

# For 3D medical image processing (NIfTI files)
pip install nibabel

pip install -U hydra-core omegaconf

pip install iopath -y

pip install webdataset -y

conda install -c conda-forge scikit-image

pip install visual-genome

# Install other dependencies
# pip install -r /local/scratch/clo37/models/Hulu-Med/requirements.txt

pip install -r /home/clo37/priv/MED-VLM-MIA/requirements.txt