#!/usr/bin/env bash
# One-time setup of the AI Vision Lab on a Jetson Orin Nano (JetPack 6.x, Ubuntu 22.04, Python 3.10).
#
#   cd ai-vision-lab/jetson && bash setup_jetson.sh
#
# Why this is different from a normal `pip install ultralytics`:
#   * pip's PyTorch is CPU-only on ARM. NVIDIA publishes CUDA builds for JetPack, so we install those.
#   * pip's opencv-python has no GStreamer, so the CSI camera (nvarguscamerasrc) would not open.
#     JetPack's system OpenCV has it; the venv sees it through --system-site-packages.
#   * Ultralytics is installed with --no-deps so it can't replace either of the above.
set -euo pipefail
cd "$(dirname "$0")"
VENV=../.venv
JP_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126     # JetPack 6.1/6.2 = CUDA 12.6

echo "== 1. System packages =="
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libopenblas-dev curl
# JetPack usually ships NVIDIA's OpenCV (with GStreamer). Only fall back to Ubuntu's
# package if there is no system OpenCV at all – the two conflict in apt.
if /usr/bin/python3 -c "import cv2" 2>/dev/null; then
    echo "System OpenCV found: $(/usr/bin/python3 -c 'import cv2; print(cv2.__version__)')"
else
    sudo apt-get install -y python3-opencv
fi

echo "== 2. Virtual environment (can see JetPack's OpenCV) =="
if [ ! -d "$VENV" ]; then
    python3 -m venv --system-site-packages "$VENV"
fi
PY="$(cd "$VENV" && pwd)/bin/python"
"$PY" -m pip install --upgrade pip wheel

echo "== 3. PyTorch with CUDA (NVIDIA build) =="
if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "CUDA PyTorch already works: $("$PY" -c 'import torch; print(torch.__version__)')"
else
    "$PY" -m pip install torch torchvision --index-url "$JP_INDEX"
fi

echo "== 4. Ultralytics without its dependencies, then the safe ones =="
# numpy<2: JetPack's OpenCV is compiled against numpy 1.x and crashes with numpy 2.
"$PY" -m pip install --no-deps ultralytics
"$PY" -m pip install "numpy<2" matplotlib pillow pyyaml requests scipy psutil polars ultralytics-thop \
                     flask waitress
# YOLO-World's text encoder (only used by the AI helper for unusual labels)
"$PY" -m pip install --no-deps "git+https://github.com/ultralytics/CLIP.git" ftfy regex wcwidth || \
    echo "   (CLIP failed to install – the AI helper will still work for trained/COCO labels)"

echo "== 5. Checks =="
"$PY" - <<'EOF'
import torch, cv2
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
gst = [l for l in cv2.getBuildInformation().splitlines() if "GStreamer" in l]
print("OpenCV", cv2.__version__, "|", gst[0].strip() if gst else "GStreamer: unknown")
assert torch.cuda.is_available(), "PyTorch can't see the GPU – training would run on the CPU"
EOF

echo "== 6. Download the model weights now (so the Lab also works offline) =="
mkdir -p ../data/models
(cd ../data/models && "$PY" - <<'EOF'
from ultralytics import YOLO
# yolo26n.pt is what Ultralytics' AMP self-check loads; training runs from this folder, so it finds it here
for w in ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo26n.pt"):
    try:
        YOLO(w); print("  ok", w)
    except Exception as e:
        print("  skipped", w, e)
try:   # YOLO-World + its CLIP text encoder (~340 MB, cached in ~/.cache/clip) for the AI helper
    YOLO("yolov8s-worldv2.pt").set_classes(["door"]); print("  ok yolov8s-worldv2.pt + CLIP")
except Exception as e:
    print("  skipped YOLO-World", e)
EOF
) || true

cat <<'EOF'

Done. Start the Lab with:
    ../.venv/bin/python run.py

Recommended for training without thermal throttling (see docs/JETSON.md):
    sudo nvpmodel -m 0          # full power mode
    sudo jetson_clocks --fan    # fan to maximum while training
EOF
