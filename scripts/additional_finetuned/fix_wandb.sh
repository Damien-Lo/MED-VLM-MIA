#!/bin/bash
#SBATCH --job-name=fix_wandb
#SBATCH --output=out_fix_wandb.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

source /local/scratch/clo37/anaconda3/etc/profile.d/conda.sh
conda activate /local/scratch/clo37/anaconda3/envs/med_vlm_mia_venv

echo "Python executable:"
which python
python -c "import sys; print(sys.executable)"

echo "Current package metadata:"
python -m pip show wandb protobuf || true

WANDB_VERSION=$(python -m pip show wandb 2>/dev/null | awk '/^Version:/ {print $2}')

if [ -z "${WANDB_VERSION}" ]; then
    echo "Could not detect existing wandb version; installing current wandb release."
else
    echo "Detected wandb version: ${WANDB_VERSION}"
fi

echo "Removing existing wandb installation..."
python -m pip uninstall -y wandb || true

rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/wandb"
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages"/wandb-*.dist-info

echo "Reinstalling wandb cleanly..."
if [ -n "${WANDB_VERSION}" ]; then
    python -m pip install --no-cache-dir --force-reinstall "wandb==${WANDB_VERSION}"
else
    python -m pip install --no-cache-dir wandb
fi

echo "Testing imports..."
python - <<'PY'
import google.protobuf
import wandb

print("protobuf:", google.protobuf.__version__)
print("wandb:", wandb.__version__)
print("wandb import successful")
PY