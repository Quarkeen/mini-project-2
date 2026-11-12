"""
Flask app: the kids' web UI + JSON API + the car's /detect endpoint.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

from . import autolabel, config, datasets, inference, models, system
from .camera import camera
from .imagestore import store
from .training import build_dir, manager, run_dir

log = logging.getLogger("lab")
STATIC = Path(__file__).parent / "static"
_car_lock = threading.Lock()


def create_app() -> Flask:
    manager.mark_interrupted()
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024      # batch uploads of phone photos

    @app.errorhandler(ValueError)
    @app.errorhandler(KeyError)
    @app.errorhandler(FileNotFoundError)
    @app.errorhandler(RuntimeError)
    def _user_error(e):
        msg = e.args[0] if e.args else str(e)
        return jsonify(error=str(msg)), 400

    @app.after_request
    def _no_cache_api(resp):
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # ── page ─────────────────────────────────────────────────────────────
    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/api/info")
    def info():
        return jsonify(
            jetson=config.IS_JETSON,
            device=config.torch_device(),
            models=config.MODEL_CATALOG,
            label_emoji=config.LABEL_EMOJI,
            camera_fps=config.CAMERA_MAX_FPS,
            port=request.host,
        )

    @app.get("/api/system")
    def sys_stats():
        s = system.stats()
        s["training"] = manager.is_running()
        return jsonify(s)

    # ── pictures ─────────────────────────────────────────────────────────
    @app.get("/api/classes")
    def classes():
        return jsonify(datasets.available_classes())

    @app.get("/api/pictures/<store_name>")
    def pictures(store_name):
        st = store(store_name)
        label = request.args.get("label")
        filt = request.args.get("filter", "all")   # all | unchecked | checked
        items = []
        for name, rec in sorted(st.items().items(), reverse=st.editable):
            labels = sorted({b[0] for b in rec.get("boxes", [])})
            if label and label not in labels and rec.get("hint") != label:
                continue
            if filt == "unchecked" and rec.get("reviewed", True):
                continue
            if filt == "checked" and not rec.get("reviewed", True):
                continue
            items.append({"name": name, "labels": labels, "boxes": rec.get("boxes", []),
                          "reviewed": rec.get("reviewed", True), "hint": rec.get("hint", ""),
                          "source": rec.get("source", "")})
        limit = int(request.args.get("limit", 120))
        offset = int(request.args.get("offset", 0))
        return jsonify(total=len(items), items=items[offset:offset + limit], editable=st.editable)

    @app.get("/img/<store_name>/<name>")
    def image(store_name, name):
        st = store(store_name)
        p = st.thumb_path(name) if request.args.get("thumb") else st.image_path(name)
        if not p.exists():
            abort(404)
        resp = send_file(p, mimetype="image/jpeg", max_age=3600)
        return resp

    @app.post("/api/mine/upload")
    def upload():
        st = store("mine")
        hint = request.form.get("label", "")
        names, errors = [], []
        for f in request.files.getlist("files"):
            try:
                names.append(st.add_image(f.read(), hint=hint, source="upload"))
            except Exception as e:
                errors.append(f"{f.filename}: {e}")
        return jsonify(names=names, errors=errors)

    @app.post("/api/mine/<name>/boxes")
    def save_boxes(name):
        body = request.get_json(force=True)
        rec = store("mine").set_boxes(name, body.get("boxes", []), body.get("reviewed"))
        return jsonify(rec)

    @app.delete("/api/mine/<name>")
    def delete_picture(name):
        store("mine").delete(name)
        return jsonify(ok=True)

    @app.post("/api/mine/<name>/suggest")
    def suggest(name):
        body = request.get_json(force=True)
        path = store("mine").image_path(name)
        return jsonify(autolabel.suggest(str(path), body.get("labels", [])))

    # ── training ─────────────────────────────────────────────────────────
    @app.post("/api/train/start")
    def train_start():
        body = request.get_json(force=True)
        manager.reserve()                 # from here on inference stays off the GPU
        try:
            camera.stop_for_training()    # the live camera would otherwise compete for memory
            inference.release_gpu()       # give training all the GPU memory
            return jsonify(manager.start(body))
        finally:
            manager.unreserve()

    @app.post("/api/train/stop")
    def train_stop():
        force = bool((request.get_json(silent=True) or {}).get("force"))
        return jsonify(stopped=manager.stop(force=force))

    @app.get("/api/train/current")
    def train_current():
        return jsonify(running=manager.is_running(), run=manager.run_id)

    @app.get("/api/runs")
    def runs():
        return jsonify(manager.runs())

    @app.get("/api/runs/<run_id>")
    def run_detail(run_id):
        r = manager.run(run_id)
        if not r:
            abort(404)
        return jsonify(r)

    @app.get("/api/runs/<run_id>/events")
    def run_events(run_id):
        since = int(request.args.get("since", 0))
        ev = manager.events(run_id, since)
        return jsonify(events=ev, next=since + len(ev),
                       running=manager.is_running() and manager.run_id == run_id)

    @app.get("/api/runs/<run_id>/log")
    def run_log(run_id):
        p = run_dir(run_id) / "train.log"
        if not p.exists():
            return Response("", mimetype="text/plain")
        text = p.read_text(errors="replace")
        return Response(text[-20000:], mimetype="text/plain")

    @app.delete("/api/runs/<run_id>")
    def run_delete(run_id):
        manager.delete(run_id)
        return jsonify(ok=True)

    @app.get("/runs/<run_id>/preview/<name>")
    def run_preview(run_id, name):
        base = (build_dir(run_id) / "images" / "val").resolve()
        p = (base / name).resolve()
        if p.parent != base or not p.exists():
            abort(404)
        return send_file(p, mimetype="image/jpeg", max_age=3600)

    # ── models & detection ───────────────────────────────────────────────
    @app.get("/api/models")
    def list_models():
        return jsonify(models=models.list_models(), active=models.active_model())

    @app.post("/api/models/active")
    def set_active():
        return jsonify(active=models.set_active_model(request.get_json(force=True).get("path")))

    @app.post("/api/detect")
    def api_detect():
        f = request.files.get("image")
        if f is None:
            raise ValueError("No picture sent.")
        img = inference.decode_image(f.read())
        model_path = models.resolve(request.form.get("model"))
        conf = float(request.form.get("conf", 0.35))
        imgsz = int(request.form.get("imgsz", 640))
        res = inference.detect(img, model_path, conf=conf, imgsz=imgsz)
        return jsonify(res)

    @app.get("/api/camera/stream")
    def camera_stream():
        camera.configure(models.resolve(request.args.get("model")),
                         float(request.args.get("conf", 0.35)))
        return Response(camera.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/camera/info")
    def camera_info():
        return jsonify(camera.info())

    # ── the car (Raspberry Pi) ───────────────────────────────────────────
    @app.get("/health")
    def health():
        try:
            path = models.active_model()
            return jsonify(status="ok", model=Path(path).name, classes=models.model_classes(path))
        except Exception as e:
            return jsonify(status="error", error=str(e)), 500

    @app.post("/detect")
    def car_detect():
        """Called by the car. Same response format as the old jetson_inference_server.py."""
        f = request.files.get("image")
        if f is None:
            return jsonify(error="No image field in request"), 400
        data = f.read()
        try:
            img = inference.decode_image(data)
        except ValueError:
            return jsonify(error="Could not decode image"), 400
        place = request.form.get("place", "")
        res = inference.detect(img, models.active_model(),
                               conf=float(request.form.get("conf", 0.35)), annotate=True)
        annotated = inference.fit(res.pop("annotated_bgr"))     # keep the reply small for the car's Wi-Fi
        _log_car_photo(data, annotated, res, place)
        return jsonify(
            labels=res["labels"],
            detections=[{"class": d["label"], "confidence": d["confidence"], "bbox": d["box"]}
                        for d in res["detections"]],
            description=res["description"],
            annotated_image=inference.to_b64_jpeg(annotated),
        )

    @app.get("/api/car/photos")
    def car_photos():
        log_path = config.CAR_DIR / "log.json"
        items = json.loads(log_path.read_text()) if log_path.exists() else []
        return jsonify(items[::-1])

    @app.get("/car/<name>")
    def car_image(name):
        base = config.CAR_DIR.resolve()
        p = (base / name).resolve()
        if p.parent != base or not p.exists():
            abort(404)
        return send_file(p, mimetype="image/jpeg")

    return app


def _log_car_photo(raw: bytes, annotated, res: dict, place: str, keep: int = 100):
    """Remember what the car saw so kids can browse it on the Car page."""
    with _car_lock:
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        (config.CAR_DIR / f"{stamp}.jpg").write_bytes(raw)
        (config.CAR_DIR / f"{stamp}_boxes.jpg").write_bytes(inference.encode_jpeg(annotated, 80))
        log_path = config.CAR_DIR / "log.json"
        items = json.loads(log_path.read_text()) if log_path.exists() else []
        items.append({"time": time.time(), "image": f"{stamp}.jpg", "annotated": f"{stamp}_boxes.jpg",
                      "description": res["description"], "labels": res["labels"], "place": place})
        for old in items[:-keep]:
            for k in ("image", "annotated"):
                p = config.CAR_DIR / old[k]
                if p.exists():
                    p.unlink()
        log_path.write_text(json.dumps(items[-keep:]))
