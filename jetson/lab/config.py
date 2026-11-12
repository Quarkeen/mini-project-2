"""
Central settings for the AI Vision Lab.

Every path is anchored to the repository, never to the current working
directory, so the app behaves the same no matter where it is launched from.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # ai-vision-lab/
DATA = Path(os.environ.get("LAB_DATA", ROOT / "data")).resolve()

LIBRARY_DIR = DATA / "library"      # built-in pictures (read-only in the UI)
MINE_DIR = DATA / "mine"            # pictures uploaded by the kids
BUILDS_DIR = DATA / "builds"        # per-run YOLO datasets assembled from the stores
RUNS_DIR = DATA / "runs"            # one folder per training run
MODELS_DIR = DATA / "models"        # pretrained weights + imported models
CAR_DIR = DATA / "car"              # photos the car sent for detection
ACTIVE_MODEL_FILE = DATA / "active_model.json"

for _d in (LIBRARY_DIR, MINE_DIR, BUILDS_DIR, RUNS_DIR, MODELS_DIR, CAR_DIR):
    _d.mkdir(parents=True, exist_ok=True)

IS_JETSON = Path("/etc/nv_tegra_release").exists() or "tegra" in platform.release().lower()

# Pictures are shrunk on import. Phone photos are 12 MP; YOLO trains at 320-640 px,
# so keeping the originals only wastes disk, RAM and Jetson CPU time decoding them.
STORE_MAX_SIDE = 1024
THUMB_SIDE = 256

# ── Jetson-friendly limits ───────────────────────────────────────────────
# The Orin Nano shares 8 GB between CPU and GPU. These caps keep training
# inside memory and leave CPU headroom so the web UI stays responsive.
TRAIN_WORKERS = 2 if IS_JETSON else 4
TRAIN_NICE = 5                       # training process runs at lower CPU priority
# Soft thermal guard: pause training between batches while the hottest
# sensor is above THERMAL_PAUSE_C, resume once it drops THERMAL_HYSTERESIS_C
# below. Keeps the board out of hardware throttling. Laptops manage their own fans.
THERMAL_GUARD = IS_JETSON
THERMAL_PAUSE_C = 85.0
THERMAL_HYSTERESIS_C = 7.0
CAMERA_MAX_FPS = 8 if IS_JETSON else 15
CAMERA_SIZE = (640, 480)

# ── Models kids can pick ─────────────────────────────────────────────────
MODEL_CATALOG = [
    {
        "id": "yolov8n", "name": "YOLOv8 Nano", "emoji": "🐭",
        "params": "3.2 M", "speed": "Fastest",
        "about": "A tiny brain. Learns quickly and runs fast on the Jetson. Best place to start.",
        "batch": {"jetson": 8, "pc": 16}, "jetson_ok": True,
    },
    {
        "id": "yolov8s", "name": "YOLOv8 Small", "emoji": "🐱",
        "params": "11.2 M", "speed": "Fast",
        "about": "A bigger brain. Usually more accurate, takes about 2-3x longer to train.",
        "batch": {"jetson": 4, "pc": 16}, "jetson_ok": True,
    },
    {
        "id": "yolov8m", "name": "YOLOv8 Medium", "emoji": "🐘",
        "params": "25.9 M", "speed": "Slow",
        "about": "A large brain. Most accurate of the three but slow to train on the Jetson.",
        "batch": {"jetson": 2, "pc": 8}, "jetson_ok": True,
    },
]
MODEL_IDS = {m["id"] for m in MODEL_CATALOG}

# Pretrained COCO model used by the "AI helper" and as the default test model.
HELPER_MODEL = "yolov8n.pt"
# Open-vocabulary model for helper suggestions on labels COCO doesn't know (door, elevator...).
WORLD_MODEL = "yolov8s-worldv2.pt"

# Label suggestions shown when kids tag their own photos.
LABEL_EMOJI = {
    "person": "🧍", "chair": "🪑", "bottle": "🧴", "cup": "☕", "book": "📚",
    "laptop": "💻", "cell phone": "📱", "backpack": "🎒", "clock": "🕰️",
    "potted plant": "🪴", "teddy bear": "🧸", "banana": "🍌", "door": "🚪",
    "elevator": "🛗", "keyboard": "⌨️", "mouse": "🖱️", "scissors": "✂️",
    "apple": "🍎", "dog": "🐶", "cat": "🐱", "umbrella": "☂️", "tv": "📺",
}

DEFAULT_PORT = 8888   # the Raspberry Pi car posts to http://<jetson>:8888/detect


def torch_device() -> str:
    """'0' when a CUDA GPU is usable, else 'cpu'."""
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
