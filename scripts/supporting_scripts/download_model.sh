#!/bin/bash
#SBATCH --job-name=download_model
#SBATCH --output=out_download_model.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G


# Load environment
source ~/.bashrc
conda activate med_vlm_mia_venv

git lfs install

cd /local/scratch/clo37/models/

cat test.txt

# git clone https://huggingface.co/ZJU-AI4H/Hulu-Med-32B
