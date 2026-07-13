#!/bin/bash
#SBATCH --job-name=install_flash_attn
#SBATCH --output=out_install_flash_attn.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G


# Load environment
source ~/.bashrc
conda activate med_vlm_mia_venv

python - <<EOF
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
EOF

which nvcc
nvcc --version
echo $CUDA_HOME


module avail cuda 2>&1 | head

# pip install flash-attn --no-build-isolation

