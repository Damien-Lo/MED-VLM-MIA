#!/bin/bash

# VLM MIA Environment Installation Script

set -e


RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'


print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Activate the environment
print_status "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Check current Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_status "Current Python version: $PYTHON_VERSION"


print_status "Updating pip..."
pip install --upgrade pip

# Check if PyTorch is already installed
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    print_warning "PyTorch is already installed (version: $TORCH_VERSION)"
    print_warning "Skipping PyTorch installation to avoid conflicts"
else
    print_status "Installing PyTorch with CUDA support..."
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
fi


cat > temp_requirements.txt << 'EOF'
accelerate==1.11.0
aiofiles==23.2.1
aiohappyeyeballs==2.6.1
aiohttp==3.12.13
aiosignal==1.3.2
altair==5.5.0
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
anyio==4.9.0
async-timeout==5.0.1
attrs==25.3.0
bitsandbytes==0.45.5
braceexpand==0.1.7
Brotli==1.1.0
certifi==2025.6.15
charset-normalizer==3.4.2
click==8.2.1
contourpy==1.3.2
cycler==0.12.1
datasets==3.6.0
decord==0.6.0
dill==0.3.8
distlib==0.3.8
einops==0.6.1
einops-exts==0.0.4
exceptiongroup==1.3.0
fastapi==0.115.14
ffmpy==0.6.0
filelock==3.19.1
fonttools==4.58.4
frozenlist==1.7.0
fsspec==2025.3.0
gitdb==4.0.12
GitPython==3.1.45
gradio==5.49.1
gradio_client==1.13.3
groovy==0.1.2
h11==0.16.0
hf-xet==1.1.5
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==0.35.3
hydra-core==1.3.2
idna==3.10
imageio==2.37.0
importlib_resources==6.5.2
intel-cmplr-lib-ur==2025.3.0
intel-openmp==2025.3.0
iopath==0.1.10
Jinja2==3.1.6
joblib==1.5.1
jsonschema==4.24.0
jsonschema-specifications==2025.4.1
kiwisolver==1.4.8
latex2mathml==3.78.0
lazy_loader==0.4
markdown-it-py==3.0.0
markdown2==2.5.3
MarkupSafe==2.1.5
matplotlib==3.10.3
mdurl==0.1.2
mkl==2025.3.0
mkl-service==2.4.0
mpmath==1.3.0
multidict==6.6.3
multiprocess==0.70.16
narwhals==1.45.0
networkx==3.4.2
numpy==1.26.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-cufile-cu12==1.11.1.6
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-cusparselt-cu12==0.6.3
nvidia-nccl-cu12==2.18.1
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvtx-cu12==12.1.105
omegaconf==2.3.0
onemkl-license==2025.3.0
opencv-python==4.7.0.72
orjson==3.10.18
packaging==25.0
pandas==2.3.0
peft==0.10.0
pillow==10.4.0
platformdirs==4.1.0
ply==3.11
portalocker==3.2.0
progressbar2==4.5.0
propcache==0.3.2
protobuf==6.31.1
psutil==7.0.0
pyarrow==20.0.0
pydantic==2.11.7
pydantic_core==2.33.2
pydub==0.25.1
Pygments==2.19.2
pyparsing==3.2.5
PyQt5==5.15.10
PyQt5-Qt5==5.15.17
PyQt5_sip==12.17.1
python-dateutil==2.9.0.post0
python-multipart==0.0.20
python-utils==3.9.1
pytz==2025.2
PyYAML==6.0.2
referencing==0.36.2
regex==2024.11.6
requests==2.32.4
rich==14.0.0
rpds-py==0.26.0
ruff==0.12.1
safehttpx==0.1.6
safetensors==0.5.3
scikit-image==0.25.2
scikit-learn==1.7.2
scipy==1.15.3
semantic-version==2.10.0
sentencepiece==0.1.99
sentry-sdk==2.42.1
shellingham==1.5.4
shortuuid==1.0.13
six==1.17.0
smmap==5.0.2
sniffio==1.3.1
starlette==0.46.2
svgwrite==1.4.3
sympy==1.14.0
tbb==2022.3.0
tcmlib==1.4.1
threadpoolctl==3.6.0
tifffile==2025.5.10
timm==0.6.13
tokenizers==0.15.1
tomlkit==0.12.0
torch==2.1.2+cu121
torchaudio==2.1.2+cu121
torchvision==0.16.2+cu121
tqdm==4.67.1
transformers==4.37.2
triton==2.1.0
typer==0.16.0
typing-inspection==0.4.1
typing_extensions==4.14.0
tzdata==2025.2
umf==1.0.2
urllib3==2.5.0
uvicorn==0.35.0
virtualenv==20.25.0
visual-genome==1.1.1
wandb==0.22.2
wavedrom==2.0.3.post3
webdataset==1.0.2
websockets==15.0.1
xxhash==3.5.0
yarl==1.20.1
EOF


pip install -r temp_requirements.txt

# Verify installation
print_status "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import gradio; print(f'Gradio version: {gradio.__version__}')"

print_success "Environment '$ENV_NAME' has been successfully created and configured!"
print_status "To activate this environment, run: conda activate $ENV_NAME"
print_status "To deactivate, run: conda deactivate"

# Display environment info
print_status "Environment information:"
conda list -n "$ENV_NAME" | head -20
print_status "... (showing first 20 packages, run 'conda list -n $ENV_NAME' to see all)"
