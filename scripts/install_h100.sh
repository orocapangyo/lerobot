#!/bin/bash
set -e

echo "🚀 H100 Server LeRobot Installation Script Starting..."

# 0. Git Clone (If not already in the repository)
if [ ! -f "pyproject.toml" ]; then
    echo "📂 Repository not found. Cloning LeRobot..."
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
else
    echo "✅ Already inside LeRobot repository."
fi

# 1. System dependencies (Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "📦 Installing system dependencies..."
    sudo apt-get update && sudo apt-get install -y \
        git \
        ffmpeg \
        libxcb-cursor0 \
        libxcb-xinerama0 \
        libxcb-xfixies0 \
        libxcb-shape0 \
        libxcb-render-util0 \
        libxcb-icccm4 \
        libxcb-keysyms1 \
        libxcb-image0
fi

# 2. Install uv (Fast dependency manager)
if ! command -v uv &> /dev/null; then
    echo "🪄 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 3. Setup Virtual Environment
echo "🐍 Setting up virtual environment..."
uv venv .venv
source .venv/bin/activate

# 4. Install LeRobot with all training-heavy extras
# extras explanation:
# - [xvla]: For VLA model training (Florence-2 based)
# - [peft]: For Parameter-Efficient Fine-Tuning (LoRA, etc.)
# - [aloha, pusht, libero]: Simulation environments for training/evaluation
# - [transformers-dep]: Vision-language model dependencies
echo "📦 Installing LeRobot and required extras..."
uv pip install -e ".[xvla,peft,aloha,pusht,libero,transformers-dep]"

# 5. Verify GPU
echo "🔍 Verifying NVIDIA GPU and CUDA..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "✅ Installation Complete!"
echo "💡 To start using your environment, run: source .venv/bin/activate"
