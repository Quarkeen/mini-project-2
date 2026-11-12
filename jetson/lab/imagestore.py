"""
A folder of pictures plus their boxes.

Layout:
    <store>/images/<name>.jpg
    <store>/thumbs/<name>.jpg          (generated on demand)
    <store>/labels.json                {name: {"boxes": [[label, cx, cy, w, h], ...],
                                               "reviewed": bool, "hint": str, "source": str}}

Boxes are stored by label *name* (not class id) and in normalised 0-1 YOLO
coordinates, so pictures from different sources can be combined freely: two
"door" boxes are always the same class.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from . import config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image_fixed(src) -> Image.Image:
    """Open an image, apply its EXIF rotation (phone photos!) and convert to RGB."""
    im = Image.open(src)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def shrink(im: Image.Image, max_side: int) -> Image.Image:
    if max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def clean_label(label: str) -> str:
    return " ".join(str(label).strip().lower().split())[:40]


def clean_boxes(boxes) -> list[list]:
    """Validate and clamp boxes coming from the browser or a model."""
    out = []
    for b in boxes or []:
        try:
            label = clean_label(b[0])
            cx, cy, w, h = (float(v) for v in b[1:5])
        except (TypeError, ValueError, IndexError):
            continue
        # clamp the box to the image
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        if not label or x2 - x1 < 0.005 or y2 - y1 < 0.005:
            continue
        out.append([label, round((x1 + x2) / 2, 6), round((y1 + y2) / 2, 6),
                    round(x2 - x1, 6), round(y2 - y1, 6)])
    return out


class ImageStore:
    def __init__(self, root: Path, name: str, editable: bool):
        self.root = Path(root)
        self.name = name
        self.editable = editable
        self.images = self.root / "images"
        self.thumbs = self.root / "thumbs"
        self.labels_path = self.root / "labels.json"
        self.images.mkdir(parents=True, exist_ok=True)
        self.thumbs.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data: dict[str, dict] = {}
        self._mtime = None
        self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self):
        with self._lock:
            if self.labels_path.exists():
                mtime = self.labels_path.stat().st_mtime
                if mtime != self._mtime:
                    self._data = json.loads(self.labels_path.read_text())
                    self._mtime = mtime
            else:
                self._data = {}

    def _save(self):
        tmp = self.labels_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=0))
        os.replace(tmp, self.labels_path)
        self._mtime = self.labels_path.stat().st_mtime

    def reload(self):
        self._load()

    # ── queries ──────────────────────────────────────────────────────────
    def items(self) -> dict[str, dict]:
        self._load()
        with self._lock:
            return {k: dict(v) for k, v in self._data.items()}

    def get(self, name: str) -> dict | None:
        self._load()
        with self._lock:
            v = self._data.get(name)
            return dict(v) if v else None

    def image_path(self, name: str) -> Path:
        p = (self.images / name).resolve()
        if p.parent != self.images.resolve() or p.name != name:
            raise ValueError("bad image name")
        return p

    def thumb_path(self, name: str) -> Path:
        src = self.image_path(name)
        dst = self.thumbs / name
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            im = shrink(load_image_fixed(src), config.THUMB_SIDE)
            im.save(dst, "JPEG", quality=80)
        return dst

    def usable(self, rec: dict) -> bool:
        """Can this picture be used for training?"""
        return bool(rec.get("boxes")) and (rec.get("reviewed", True) or not self.editable)

    def class_counts(self, only_usable: bool = True) -> Counter:
        """How many pictures contain each label."""
        c: Counter = Counter()
        for rec in self.items().values():
            if only_usable and not self.usable(rec):
                continue
            for label in {b[0] for b in rec.get("boxes", [])}:
                c[label] += 1
        return c

    # ── edits ────────────────────────────────────────────────────────────
    def add_image(self, data: bytes | Path, hint: str = "", source: str = "upload",
                  boxes=None, reviewed: bool = False, name: str | None = None) -> str:
        src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
        im = shrink(load_image_fixed(src), config.STORE_MAX_SIDE)
        name = name or f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
        im.save(self.images / name, "JPEG", quality=90)
        with self._lock:
            self._load()
            self._data[name] = {
                "boxes": clean_boxes(boxes),
                "reviewed": reviewed,
                "hint": clean_label(hint) if hint else "",
                "source": source,
                "size": list(im.size),
            }
            self._save()
        return name

    def set_boxes(self, name: str, boxes, reviewed: bool | None = None) -> dict:
        with self._lock:
            self._load()
            rec = self._data.get(name)
            if rec is None:
                raise KeyError(name)
            rec["boxes"] = clean_boxes(boxes)
            if reviewed is not None:
                rec["reviewed"] = bool(reviewed)
            self._save()
            return dict(rec)

    def bulk_update(self, updates: dict[str, dict]):
        """Write many records at once (used by import scripts)."""
        with self._lock:
            self._load()
            self._data.update(updates)
            self._save()

    def delete(self, name: str):
        img = self.image_path(name)          # validates the name (no '..' or '/')
        with self._lock:
            self._load()
            self._data.pop(name, None)
            self._save()
        for p in (img, self.thumbs / img.name):
            if p.is_file():
                p.unlink()


_stores: dict[str, ImageStore] = {}


def store(name: str) -> ImageStore:
    if name not in _stores:
        if name == "library":
            _stores[name] = ImageStore(config.LIBRARY_DIR, "library", editable=False)
        elif name == "mine":
            _stores[name] = ImageStore(config.MINE_DIR, "mine", editable=True)
        else:
            raise KeyError(name)
    return _stores[name]
