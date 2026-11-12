"""
Build the built-in picture library kids choose from.

  * Everyday objects come from COCO 2017 (real, human-drawn bounding boxes).
    Only the YOLO label files (46 MB) are downloaded in full; images are fetched
    one by one for the classes we want, then shrunk to 640 px.
  * Door / elevator come from the project's own Roboflow-based dataset
    (data/raw/merged_door_elevator). Pictures whose label is the old fake
    "whole picture" box (0.5 0.5 0.9 0.9) are skipped.

Run once on a PC with internet (takes a few minutes), then copy data/library to the Jetson:

    python scripts/build_library.py                       # default classes, 150 pictures each
    python scripts/build_library.py --per-class 80 --classes chair bottle cup
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import config  # noqa: E402
from lab.imagestore import ImageStore, load_image_fixed, shrink  # noqa: E402

COCO_LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
COCO_IMG_URL = "http://images.cocodataset.org/train2017/{}.jpg"
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
DEFAULT_CLASSES = ["person", "chair", "bottle", "cup", "book", "laptop", "cell phone",
                   "backpack", "clock", "potted plant", "teddy bear", "banana"]


def get_labels_zip(cache: Path) -> Path:
    z = cache / "coco2017labels.zip"
    if not z.exists():
        print("Downloading COCO label files (46 MB)…")
        urllib.request.urlretrieve(COCO_LABELS_URL, z)
    return z


def read_coco_labels(zpath: Path, wanted: set[int]) -> dict[str, list]:
    """stem -> list of (cls, cx, cy, w, h) for images that contain any wanted class."""
    out = {}
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            n = info.filename
            if "/labels/train2017/" not in n or not n.endswith(".txt"):
                continue
            rows = []
            for line in z.read(info).decode().splitlines():
                p = line.split()
                if len(p) == 5:
                    rows.append((int(p[0]), *map(float, p[1:])))
            if any(r[0] in wanted for r in rows):
                out[Path(n).stem] = rows
    return out


def pick_images(labels: dict[str, list], cls_id: int, n: int, rng: random.Random) -> list[str]:
    """Images where the class is reasonably visible: biggest box >= 3% of the image, not too crowded."""
    good = []
    for stem, rows in labels.items():
        areas = [r[3] * r[4] for r in rows if r[0] == cls_id]
        if areas and max(areas) >= 0.03 and len(rows) <= 15:
            good.append(stem)
    rng.shuffle(good)
    return good[:n]


def fetch(stem: str) -> bytes:
    with urllib.request.urlopen(COCO_IMG_URL.format(stem), timeout=30) as r:
        return r.read()


def import_coco(lib: ImageStore, classes: list[str], per_class: int, cache: Path, seed: int):
    ids = {c: COCO_NAMES.index(c) for c in classes}
    keep_ids = {COCO_NAMES.index(c) for c in DEFAULT_CLASSES} | set(ids.values())
    labels = read_coco_labels(get_labels_zip(cache), set(ids.values()))
    rng = random.Random(seed)
    existing = lib.items()

    todo: dict[str, list] = {}
    for c, cid in ids.items():
        have = sum(1 for r in existing.values() if r.get("source") == "coco" and c in {b[0] for b in r["boxes"]})
        need = max(0, per_class - have)
        for stem in pick_images(labels, cid, need * 2, rng):
            if f"coco_{stem}.jpg" in existing or stem in todo:
                continue
            todo[stem] = labels[stem]
            need -= 1
            if need <= 0:
                break
    print(f"Fetching {len(todo)} COCO pictures…")

    updates, done = {}, 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fetch, stem): stem for stem in todo}
        for fut in as_completed(futs):
            stem = futs[fut]
            try:
                im = shrink(load_image_fixed(io.BytesIO(fut.result())), 640)
            except Exception as e:
                print("  skip", stem, e)
                continue
            name = f"coco_{stem}.jpg"
            im.save(lib.images / name, "JPEG", quality=88)
            # keep boxes for every library object in the picture, so an unlabelled
            # chair in a "cup" picture doesn't teach the model that chairs are background
            boxes = [[COCO_NAMES[r[0]], *r[1:]] for r in todo[stem] if r[0] in keep_ids]
            updates[name] = {"boxes": boxes, "reviewed": True, "source": "coco", "size": list(im.size)}
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)}")
                lib.bulk_update(updates)
                updates = {}
    lib.bulk_update(updates)


def import_door_elevator(lib: ImageStore, src: Path):
    """Project's own merged dataset: class 0 = elevator, 1 = door. Skips the fake full-image boxes."""
    if not src.exists():
        print("No door/elevator dataset at", src)
        return
    names = {0: "elevator", 1: "door"}
    updates, skipped = {}, 0
    for split in ("train", "val"):
        for img in sorted((src / "images" / split).glob("*")):
            lbl = src / "labels" / split / (img.stem + ".txt")
            if not lbl.exists():
                continue
            boxes = []
            fake = False
            for line in lbl.read_text().split("\n"):
                p = line.split()
                if len(p) != 5:
                    continue
                cx, cy, w, h = map(float, p[1:])
                if abs(cx - 0.5) < 1e-6 and abs(cy - 0.5) < 1e-6 and abs(w - 0.9) < 1e-6 and abs(h - 0.9) < 1e-6:
                    fake = True
                boxes.append([names[int(p[0])], cx, cy, w, h])
            if fake or not boxes:
                skipped += 1
                continue
            name = f"doorlift_{split}_{img.stem}.jpg"
            im = shrink(load_image_fixed(img), 640)
            im.save(lib.images / name, "JPEG", quality=88)
            updates[name] = {"boxes": boxes, "reviewed": True, "source": "roboflow", "size": list(im.size)}
    lib.bulk_update(updates)
    print(f"Door/elevator: imported {len(updates)} pictures, skipped {skipped} with fake whole-image boxes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-coco", action="store_true")
    args = ap.parse_args()

    bad = [c for c in args.classes if c not in COCO_NAMES]
    if bad:
        sys.exit(f"Not COCO classes: {bad}")
    cache = config.DATA / "cache"
    cache.mkdir(exist_ok=True)
    lib = ImageStore(config.LIBRARY_DIR, "library", editable=False)
    if not args.skip_coco:
        import_coco(lib, args.classes, args.per_class, cache, args.seed)
    import_door_elevator(lib, config.DATA / "raw" / "merged_door_elevator")
    counts = lib.class_counts()
    print("\nLibrary now has:")
    for label, n in sorted(counts.items()):
        print(f"  {label:14s} {n} pictures")


if __name__ == "__main__":
    main()
