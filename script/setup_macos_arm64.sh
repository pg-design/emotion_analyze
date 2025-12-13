#!/usr/bin/env bash
# scripts/setup_macos_arm64.sh
# One-shot environment setup for macOS Apple Silicon (M1/M2/M3).
# Usage: bash scripts/setup_macos_arm64.sh

set -euo pipefail

echo "🔧 Creating virtual environment .venv ..."
python3 -m venv .venv
source .venv/bin/activate

echo "⬆️  Upgrading pip/setuptools/wheel ..."
python -m pip install --upgrade pip setuptools wheel

echo "📦 Installing base numeric stack (TF 2.13 friendly) ..."
pip install "numpy==1.24.3" "pandas==2.1.4" "protobuf<5"

echo "🖼️  Installing vision/IO libs ..."
pip install "opencv-python-headless==4.9.0.80" "Pillow>=9.5" "openpyxl>=3.1.2" "tqdm>=4.66"

echo "🧠 Installing TensorFlow for macOS + Metal (needed by DeepFace) ..."
pip install "tensorflow-macos==2.13.0" "tensorflow-metal==1.0.*"

echo "🔥 Installing PyTorch CPU wheels (stable on macOS ARM) ..."
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2

echo "🤖 Installing model/tooling libraries ..."
pip install "timm==0.9.16" "facenet-pytorch==2.5.3" "deepface==0.0.96" "emotiefflib==0.2.2"

echo "🧪 Quick import test ..."
python - <<'PY'
print("✅ Environment looks OK")
import sys; print("Python:", sys.version.split()[0])
import numpy as np, pandas as pd, cv2
print("NumPy:", np.__version__, "| Pandas:", pd.__version__, "| OpenCV:", cv2.__version__)
import tensorflow as tf; print("TensorFlow:", tf.__version__)
import torch; print("Torch:", torch.__version__)
from deepface import DeepFace; print("DeepFace: OK")
from facenet_pytorch import MTCNN; print("MTCNN: OK")
from emotiefflib.facial_analysis import EmotiEffLibRecognizer; print("EmotiEffLib: OK")
PY

echo ""
echo "🎉 Done. To use this env in a new terminal:"
echo "   source .venv/bin/activate"
