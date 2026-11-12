"""
Assemble a YOLO training dataset from the picture stores.

A kid picks objects ("chair", "door"...) and how many pictures per object.
We pick pictures containing each object, keep only the boxes of the chosen
objects, split train/val per object (so every object is in both splits),
and write a standard Ultralytics layout:

    builds/<run_id>/images/{train,val}/*.jpg   (hard links, no extra disk)
    builds/<run_id>/labels/{train,val}/*.txt
    builds/<run_id>/dataset.yaml
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import yaml

from . import config
from .imagestore import store


class DatasetError(ValueError):
    pass


def available_classes() -> list[dict]:
    """Every label that has trainable pictures, with counts per store."""
    lib = store("library").class_counts()
    mine = store("mine").class_counts()
    mine_pending = store("mine").class_counts(only_usable=False)
    labels = sorted(set(lib) | set(mine) | set(mine_pending))
    return [{
        "label": label,
        "emoji": config.LABEL_EMOJI.get(label, "🔷"),
        "library": lib.get(label, 0),
        "mine": mine.get(label, 0),
        "mine_unchecked": mine_pending.get(label, 0) - mine.get(label, 0),
    } for label in labels]


def _link(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(run_id: str, classes: list[str], per_class: int, use_library: bool = True,
          use_mine: bool = True, val_ratio: float = 0.2, seed: int = 0) -> dict:
    classes = [c for c in dict.fromkeys(classes) if c]
    if not classes:
        raise DatasetError("Pick at least one object to learn.")
    rng = random.Random(seed)

    # candidates: (store_name, image_name, record)
    pools = []
    if use_library:
        pools.append(store("library"))
    if use_mine:
        pools.append(store("mine"))
    if not pools:
        raise DatasetError("Choose library pictures, your own pictures, or both.")

    by_class: dict[str, list[tuple]] = {c: [] for c in classes}
    for st in pools:
        for name, rec in st.items().items():
            if not st.usable(rec):
                continue
            present = {b[0] for b in rec["boxes"]}
            for c in classes:
                if c in present:
                    by_class[c].append((st, name, rec))

    missing = [c for c, items in by_class.items() if len(items) < 2]
    if missing:
        raise DatasetError(
            "Not enough pictures with boxes for: " + ", ".join(missing)
            + ". Each object needs at least 2 pictures (add some in My Pictures).")

    # Pick per class; split each class's pick into train/val so both splits see every class.
    # A picture can contain several chosen objects, so the first split it lands in wins.
    split_of: dict[tuple, str] = {}
    chosen: dict[tuple, tuple] = {}
    for c in classes:
        items = by_class[c][:]
        rng.shuffle(items)
        # Prefer pictures we haven't already taken for another class
        items.sort(key=lambda it: (it[0].name, it[1]) in chosen)
        items = items[:per_class]
        n_val = max(1, round(len(items) * val_ratio))
        for i, it in enumerate(items):
            key = (it[0].name, it[1])
            if key not in split_of:
                split_of[key] = "val" if i < n_val else "train"
                chosen[key] = it

    # A class whose pictures were all taken by other classes can end up in only one split:
    # move its least-shared picture across so every class is both practised and quizzed.
    chosen_set = set(classes)

    def labels_of(key):
        return {b[0] for b in chosen[key][2]["boxes"]} & chosen_set

    for c in classes:
        keys = [k for k in chosen if c in labels_of(k)]
        for want, other in (("val", "train"), ("train", "val")):
            if len(keys) >= 2 and not any(split_of[k] == want for k in keys):
                movable = sorted((k for k in keys if split_of[k] == other), key=lambda k: len(labels_of(k)))
                split_of[movable[0]] = want

    out = config.BUILDS_DIR / run_id
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    cls_index = {c: i for i, c in enumerate(classes)}
    stats = {c: {"train": 0, "val": 0, "boxes": 0} for c in classes}
    val_examples: dict[str, str] = {}

    for (st_name, img_name), (st, _, rec) in chosen.items():
        split = split_of[(st_name, img_name)]
        stem = f"{st_name}_{Path(img_name).stem}"
        dst_img = out / "images" / split / f"{stem}.jpg"
        _link(st.image_path(img_name), dst_img)
        lines = []
        seen = set()
        for label, cx, cy, w, h in rec["boxes"]:
            if label in cls_index:
                lines.append(f"{cls_index[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats[label]["boxes"] += 1
                seen.add(label)
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        for label in seen:
            stats[label][split] += 1
            if split == "val" and label not in val_examples:
                val_examples[label] = str(dst_img)

    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "names": {i: c for i, c in enumerate(classes)},
    }, sort_keys=False))

    n_train = sum(1 for v in split_of.values() if v == "train")
    n_val = len(split_of) - n_train
    # Preview strip: one validation picture per class (max 4)
    previews = list(val_examples.values())[:4]
    # top up to 4 so a one- or two-object run still shows several quiz pictures
    for p in sorted((out / "images" / "val").glob("*.jpg")):
        if len(previews) >= 4:
            break
        if str(p) not in previews:
            previews.append(str(p))
    return {"yaml": str(yaml_path), "classes": classes, "train": n_train, "val": n_val,
            "per_class": stats, "previews": previews}
