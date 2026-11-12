"""
Model files: pretrained weights, trained runs, and the "active" model the car uses.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from . import config

_dl_lock = threading.Lock()


def weights_path(filename: str) -> Path:
    """Path to a pretrained weight file inside data/models, downloading it on first use."""
    path = config.MODELS_DIR / filename
    if not path.exists():
        with _dl_lock:
            if not path.exists():
                from ultralytics.utils.downloads import attempt_download_asset
                attempt_download_asset(str(path))
        if not path.exists():
            raise FileNotFoundError(
                f"Could not download {filename}. Connect to the internet once, or copy it to {path}.")
    return path


def list_models() -> list[dict]:
    """Every model a kid can test with: the pretrained COCO brain, imported and trained models."""
    out = [{
        "path": str(config.MODELS_DIR / config.HELPER_MODEL),
        "name": "Pretrained brain (knows 80 everyday things)",
        "kind": "pretrained",
    }]
    for p in sorted(config.MODELS_DIR.glob("*.pt")):
        if p.name.startswith(("yolov8", "yolo11", "yolo26")):
            continue        # stock weights
        out.append({"path": str(p), "name": p.stem.replace("_", " "), "kind": "imported"})
    for run_json in sorted(config.RUNS_DIR.glob("*/run.json"), reverse=True):
        try:
            run = json.loads(run_json.read_text())
        except (OSError, ValueError):
            continue
        best = run_json.parent / "weights" / "best.pt"
        if best.exists():
            out.append({
                "path": str(best), "name": run.get("name") or run_json.parent.name,
                "kind": "trained", "run": run_json.parent.name,
                "classes": run.get("classes", []), "map50": (run.get("final") or {}).get("map50"),
                "state": run.get("state"),
            })
    return out


def resolve(path: str | None) -> str:
    """Validate a model path from the browser: it must live inside data/."""
    if not path:
        return active_model()
    p = Path(path).resolve()
    if config.DATA not in p.parents or p.suffix != ".pt":
        raise ValueError("Unknown model")
    if p.name == config.HELPER_MODEL and p.parent == config.MODELS_DIR:
        return str(weights_path(config.HELPER_MODEL))
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p.name}")
    return str(p)


_classes_cache: dict[str, list] = {}


def model_classes(path: str) -> list[str]:
    """Class names stored inside a .pt checkpoint (cheap: no model build)."""
    if path not in _classes_cache:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        names = (ckpt.get("model") or ckpt.get("ema")).names
        _classes_cache[path] = list(names.values()) if isinstance(names, dict) else list(names)
    return _classes_cache[path]


def active_model() -> str:
    try:
        p = Path(json.loads(config.ACTIVE_MODEL_FILE.read_text())["path"])
        if not p.is_absolute():          # stored relative to data/ so it survives copying data/
            p = config.DATA / p
        if p.exists():
            return str(p)
    except (OSError, ValueError, KeyError):
        pass
    return str(weights_path(config.HELPER_MODEL))


def set_active_model(path: str) -> str:
    p = resolve(path)
    config.ACTIVE_MODEL_FILE.write_text(json.dumps({"path": str(Path(p).relative_to(config.DATA))}))
    return p
