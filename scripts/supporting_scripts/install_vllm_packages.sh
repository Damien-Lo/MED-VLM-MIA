#!/bin/bash
#SBATCH --job-name=einstall_vllm_packages
#SBATCH --output=out_install_vllm_packages.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate med_vlm_mia_venv



# Make sure CUDA toolkit is set up
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Force Hopper H200 arch
export TORCH_CUDA_ARCH_LIST="9.0a"

# Avoid isolated build env surprises
pip install -U setuptools wheel setuptools-scm cmake ninja
pip install --no-build-isolation -v git+https://github.com/jiangsongtao/vllm.git

