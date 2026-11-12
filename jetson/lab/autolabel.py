"""
"AI helper" box suggestions for pictures kids upload.

The helper only *suggests*: boxes come back unreviewed and the kid moves,
fixes or deletes them before the picture is used for training.

For each label the helper picks the best brain that knows it:
  1. a model the kids trained that knows the label (highest mAP wins) –
     the more you train, the better your helper gets;
  2. the pretrained COCO model (chair, bottle, person, …);
  3. YOLO-World, an open-vocabulary detector that can be asked for any word.
     It is weak on things like "door", so its boxes are flagged as unsure.
"""
from __future__ import annotations

from . import config, inference, models
from .imagestore import clean_label

_world = {"model": None, "classes": None}
_yolo_cache: dict[str, object] = {}


def clear_cache():
    """Called (under inference._lock) before training starts."""
    _yolo_cache.clear()
    _world.update(model=None, classes=None)


def _yolo(path: str):
    from ultralytics import YOLO
    if path not in _yolo_cache:
        if len(_yolo_cache) >= 3:          # keep memory bounded on the Jetson
            _yolo_cache.pop(next(iter(_yolo_cache)))
        _yolo_cache[path] = YOLO(path)
    return _yolo_cache[path]


def _world_model(classes: list[str]):
    from ultralytics import YOLO
    if _world["model"] is None:
        _world["model"] = YOLO(str(models.weights_path(config.WORLD_MODEL)))
    if _world["classes"] != classes:
        _world["model"].set_classes(classes)
        _world["classes"] = classes
    return _world["model"]


def _pick_helpers(labels: list[str]) -> dict[str, list[str]]:
    """label -> helper model path (or 'world')."""
    coco_path = str(models.weights_path(config.HELPER_MODEL))
    coco_names = set(models.model_classes(coco_path))
    trained = []
    for m in models.list_models():
        if m["kind"] == "trained" and m.get("state") == "done":
            trained.append((m.get("map50") or 0.0, m["path"], set(m.get("classes") or [])))
        elif m["kind"] == "imported":
            try:
                trained.append((0.0, m["path"], set(models.model_classes(m["path"]))))
            except Exception:
                pass
    trained.sort(key=lambda t: -t[0])
    plan: dict[str, list[str]] = {}
    for label in labels:
        path = next((p for _, p, cls in trained if label in cls), None)
        if path is None:
            path = coco_path if label in coco_names else "world"
        plan.setdefault(path, []).append(label)
    return plan


def suggest(image_path: str, labels: list[str], conf: float = 0.25) -> dict:
    labels = list(dict.fromkeys(clean_label(x) for x in labels if clean_label(x)))
    if not labels:
        raise ValueError("Tell the helper which object to look for.")
    boxes, used, warning = [], [], None

    def collect(result, keep, unsure=False, top_only=False):
        h, w = result.orig_shape
        found = []
        for b in result.boxes:
            name = result.names[int(b.cls)]
            if name in keep:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                found.append([name, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h,
                              round(float(b.conf), 3), unsure])
        if top_only:   # weak detector: only its single best guess per label
            best = {}
            for f in found:
                if f[0] not in best or f[5] > best[f[0]][5]:
                    best[f[0]] = f
            found = list(best.values())
        boxes.extend(found)

    with inference._lock:
        device = inference.current_device()
        for helper, wanted in _pick_helpers(labels).items():
            if helper == "world":
                try:
                    r = _world_model(wanted).predict(image_path, conf=0.005, device=device, verbose=False)[0]
                    collect(r, set(wanted), unsure=True, top_only=True)
                    used.append("YOLO-World (unsure)")
                    warning = (f"The helper isn't good at finding {', '.join(wanted)} yet – check its boxes "
                               "carefully. Train a model on these objects and it will get better!")
                except Exception as e:
                    warning = f"The helper can't look for {', '.join(wanted)} right now ({e}). Draw the boxes yourself."
            else:
                r = _yolo(helper).predict(image_path, conf=conf, device=device, verbose=False)[0]
                collect(r, set(wanted))
                used.append("pretrained YOLOv8" if helper.endswith(config.HELPER_MODEL) else "your trained model")

    boxes.sort(key=lambda b: -b[5])
    per, out = {}, []
    for b in boxes:
        per[b[0]] = per.get(b[0], 0) + 1
        if per[b[0]] <= 10:
            out.append(b)
    res = {"boxes": out, "used": used}
    if warning:
        res["warning"] = warning
    return res
