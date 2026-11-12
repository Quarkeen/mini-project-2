"""
Object detection for the Test page, the live camera and the car.

* Up to two models stay loaded (e.g. the car's and the Test page's).
* Calls are serialised with a lock (YOLO predictors are not thread-safe and
  the Jetson has one GPU anyway).
* While a model is training, inference falls back to the CPU so it never
  competes with training for GPU memory.
"""
from __future__ import annotations

import base64
import io
import threading
from collections import Counter, OrderedDict

import cv2
import numpy as np

from . import config
from .training import manager

_lock = threading.Lock()
# Small LRU of loaded models, keyed by (path, device). Two slots so the car's model and the
# model on the Test page / camera don't evict each other on every request.
_MAX_MODELS = 2
_models: OrderedDict = OrderedDict()


def current_device() -> str:
    if manager.is_busy():
        return "cpu"
    return config.torch_device()


def _get_model(path: str, device: str):
    key = (path, device)
    if key in _models:
        _models.move_to_end(key)
        return _models[key]
    from ultralytics import YOLO
    m = YOLO(path)            # may raise (corrupt file): the cache is only touched after success
    # warm-up so the first real request isn't slow
    m.predict(np.zeros((320, 320, 3), np.uint8), device=device, verbose=False)
    _models[key] = m
    while len(_models) > _MAX_MODELS:
        _models.popitem(last=False)
        _empty_cuda()
    return m


def _empty_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_gpu():
    """Drop every cached model before training starts, so training gets all the GPU memory."""
    from . import autolabel
    with _lock:
        _models.clear()
        autolabel.clear_cache()
        _empty_cuda()


def decode_image(data: bytes) -> np.ndarray:
    """Bytes -> BGR array, upright. Phone photos are often stored sideways with an EXIF
    'rotate me' flag; older OpenCV builds (e.g. JetPack's) ignore it, so use Pillow."""
    try:
        from .imagestore import load_image_fixed
        rgb = np.asarray(load_image_fixed(io.BytesIO(data)))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not read that picture.")
        return img


def detect(img_bgr: np.ndarray, model_path: str, conf: float = 0.35, imgsz: int = 640,
           annotate: bool = False) -> dict:
    """Run the model. Boxes are returned in pixels and normalised (0-1)."""
    with _lock:
        device = current_device()
        model = _get_model(model_path, device)
        r = model.predict(img_bgr, conf=conf, imgsz=imgsz, device=device, verbose=False, max_det=50)[0]
        h, w = r.orig_shape
        dets = []
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            dets.append({
                "label": r.names[int(b.cls)],
                "confidence": round(float(b.conf), 3),
                "box": [round(x1), round(y1), round(x2), round(y2)],
                "norm": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
            })
        out = {
            "detections": dets,
            "labels": [d["label"] for d in dets],
            "description": describe([d["label"] for d in dets]),
            "width": w, "height": h, "device": device,
            "speed_ms": round(sum(r.speed.values()), 1),
            "classes": list(r.names.values()),
        }
        if annotate:
            out["annotated_bgr"] = r.plot()
        return out


def fit(img_bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img_bgr
    return cv2.resize(img_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def encode_jpeg(img_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def to_b64_jpeg(img_bgr: np.ndarray) -> str:
    # r.plot() returns BGR and cv2.imencode expects BGR, so no colour conversion (the old blue-tint bug)
    return base64.b64encode(encode_jpeg(img_bgr)).decode()


_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
_IRREGULAR = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep",
              "cell phone": "cell phones", "bus": "buses", "glass": "glasses", "bench": "benches"}


def plural(name: str, n: int) -> str:
    if n == 1:
        return name
    if name in _IRREGULAR:
        return _IRREGULAR[name]
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"


def describe(labels: list[str]) -> str:
    """['person', 'chair', 'chair'] -> 'I found one person and two chairs.'"""
    if not labels:
        return "I don't see anything I recognise."
    counts = Counter(labels)
    parts = [f"{_NUM.get(n, str(n))} {plural(name, n)}" for name, n in counts.most_common()]
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"I found {joined}."
