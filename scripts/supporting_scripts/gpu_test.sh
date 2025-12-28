#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=out_gpu_test.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G

echo "Running nvidia-smi"
nvidia-smi

echo "nvcc"
nvcc --version

echo "Toolchain"
gcc --version
cmake --version
ninja --version


echo "Torch View"
python -c "import torch; print(torch.__version__); print('cuda available:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda)"

echo "Load NVCC"
module avail cuda
module load cuda/12.4   # or whatever exists (12.4, 12.3, 12.2...)
which nvcc
nvcc --version
