#!/usr/bin/env bash
# scripts/setup_macos_arm64.sh
# One-shot environment setup for macOS Apple Silicon (M1/M2/M3).
# Usage: bash scripts/setup_macos_arm64.sh

set -euo pipefail

echo "🔧 Creating virtual environment .venv ..."
python3.10 -m venv .venv
source .venv/bin/activate

echo "⬆️  Upgrading pip/setuptools/wheel ..."
python -m pip install --upgrade pip setuptools wheel

echo "📦 Installing dependencies from requirements.txt ..."
pip install -r requirements.txt

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
