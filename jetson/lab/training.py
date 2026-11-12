"""
Training manager.

Training runs in a separate *process* (not a thread):
  * the web server stays responsive (no GIL fight, lower CPU priority),
  * CUDA memory is fully released when training ends,
  * a crash / out-of-memory kill can't take the UI down with it.

The worker streams progress as JSON lines into runs/<id>/events.jsonl,
which the browser polls. Event types:
  start   – epochs, batches per epoch, classes, device
  batch   – running train loss inside the current epoch (throttled)
  epoch   – train/val loss, precision, recall, mAP, learning rate,
            how much each part of the network changed, ETA
  preview – the model's boxes on a few fixed validation pictures
  status  – human-readable messages (thermal pause, saving...)
  done / stopped / error
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import config, datasets

# ── shared helpers ───────────────────────────────────────────────────────


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


_RUN_ID = re.compile(r"^[0-9A-Za-z_-]{1,64}$")


def run_dir(run_id: str) -> Path:
    """Folder of a run. Rejects anything that isn't a plain run id ('..', '/', '%2E%2E'...)."""
    if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
        raise ValueError("Unknown training run")
    return config.RUNS_DIR / run_id


def build_dir(run_id: str) -> Path:
    run_dir(run_id)        # same validation
    return config.BUILDS_DIR / run_id


def _write_json(p: Path, data: dict):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, p)


class _Events:
    def __init__(self, path: Path):
        self._f = open(path, "a", buffering=1)

    def emit(self, type_: str, **kw):
        kw["type"] = type_
        kw["t"] = round(time.time(), 2)
        self._f.write(json.dumps(kw) + "\n")


def _f(x, nd=5):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


# On the Jetson (unified memory) PyTorch often reports "out of memory" as an NVML assert or an
# NvMap allocation failure instead of a normal CUDA OOM, so Ultralytics' own batch-halving never
# triggers. Recognise all of them.
_OOM_MARKERS = ("out of memory", "nvml_success", "cudacachingallocator", "nvmap",
                "cublas_status_alloc_failed", "cudnn_status_alloc_failed", "cuda error: unknown error")


def _is_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return any(m in msg for m in _OOM_MARKERS)


def _jetson_dataloader_tweaks():
    """The Jetson's CPU and GPU share the same RAM, so "pinned" (page-locked) memory buys nothing
    there – and Ultralytics always pins every batch. The pinned pool ran out a few epochs into
    training ("CUDA error: out of memory" in the pin-memory thread during validation).
    Turn pinning off and buffer fewer batches per loader worker."""
    from ultralytics.data import build

    orig_init = build.InfiniteDataLoader.__init__
    if getattr(orig_init, "_lab_patched", False):
        return

    def init(self, *args, **kwargs):
        kwargs["pin_memory"] = False
        if kwargs.get("num_workers") and kwargs.get("prefetch_factor"):
            kwargs["prefetch_factor"] = 2
        orig_init(self, *args, **kwargs)

    init._lab_patched = True
    build.InfiniteDataLoader.__init__ = init


def _lr_of(tr):
    """Learning rate of the weights. Ultralytics has 3 optimizer groups (weights, norm layers,
    biases) and the biases warm up from a much higher value; their order differs between
    versions (8.3 vs 8.4), so take the median instead of trusting 'pg0'."""
    vals = sorted(v for v in (getattr(tr, "lr", None) or {}).values() if isinstance(v, (int, float)))
    return vals[len(vals) // 2] if vals else None


def _losses(tl) -> dict:
    """Running train loss as {box, cls, dfl}. Newer Ultralytics keeps a dict, older ones a tensor."""
    if tl is None:
        return {}
    if isinstance(tl, dict):
        return {str(k).replace("_loss", "").split("/")[-1]: _f(v) for k, v in tl.items()}
    vals = tl.tolist() if hasattr(tl, "tolist") else [tl]
    if not isinstance(vals, list):
        vals = [vals]
    return dict(zip(("box", "cls", "dfl"), (_f(v) for v in vals)))


# ── the worker process ───────────────────────────────────────────────────

def _backbone_end(net) -> int:
    """Index of the last backbone layer: SPPF (YOLOv8), or the C2PSA attention block right
    after it (YOLO11)."""
    layers = list(net.model)
    end = max((i for i, m in enumerate(layers) if type(m).__name__ == "SPPF"), default=len(layers) // 2)
    if end + 1 < len(layers) and type(layers[end + 1]).__name__ == "C2PSA":
        end += 1
    return end


def _param_groups(net):
    """Split YOLO layers into the three parts we show kids:
    backbone ("eyes": finds edges/shapes), neck ("mixer": combines sizes), head ("decider": boxes + labels)."""
    layers = list(net.model)
    backbone_end = _backbone_end(net)
    head_start = len(layers) - 1
    groups = {}
    for name, p in net.named_parameters():
        parts = name.split(".")
        try:
            idx = int(parts[1])
        except (IndexError, ValueError):
            continue
        g = "backbone" if idx <= backbone_end else ("head" if idx >= head_start else "neck")
        groups.setdefault(g, []).append(name)
    return groups


def _snapshot(net):
    """Copy of the learnable weights only. Normalisation running statistics are left out: they
    keep updating even in frozen layers, which would make a frozen backbone look like it learned."""
    import torch
    with torch.no_grad():
        return {k: v.detach().to("cpu", torch.float16, copy=True) for k, v in net.named_parameters()}


def _change(groups, cur, ref) -> dict:
    """Relative change (%) of each part's weights between two snapshots."""
    import torch
    out = {}
    for g, names in groups.items():
        num = den = 0.0
        for n in names:
            if n in cur and n in ref:
                a, b = cur[n].float(), ref[n].float()
                num += torch.sum((a - b) ** 2).item()
                den += torch.sum(b ** 2).item()
        out[g] = round(100.0 * (num ** 0.5) / (den ** 0.5 + 1e-12), 4)
    return out


def _worker(run_id: str, cfg: dict, stop_event):
    run_dir = config.RUNS_DIR / run_id
    log = open(run_dir / "train.log", "a", buffering=1)
    sys.stdout = sys.stderr = log
    os.chdir(config.MODELS_DIR)          # any stray Ultralytics download lands in data/models
    if sys.platform.startswith("linux"):
        # This process was spawned, so it inherits "spawn" as the default start method; the
        # DataLoader's spawned workers then die on startup and training hangs. Fork, as in a plain script.
        mp.set_start_method("fork", force=True)
    try:
        os.nice(config.TRAIN_NICE)
    except (AttributeError, OSError):
        pass
    ev = _Events(run_dir / "events.jsonl")
    run = _read_json(run_dir / "run.json")

    def set_state(state, **kw):
        run.update(state=state, **kw)
        _write_json(run_dir / "run.json", run)

    try:
        import torch
        from ultralytics import YOLO

        from .models import weights_path
        from .system import ThermalGuard

        if config.IS_JETSON:
            _jetson_dataloader_tweaks()

        device = config.torch_device()
        ev.emit("status", msg=f"Loading {cfg['model']} brain on {'GPU' if device != 'cpu' else 'CPU'}…")

        previews = cfg.get("previews", [])
        st = {"in_epoch": False, "batch": 0, "last_emit": 0.0, "epoch_t0": time.time(),
              "epoch_times": [], "nb": 0, "groups": None, "start_w": None, "prev_w": None}

        guard = ThermalGuard(
            should_abort=stop_event.is_set,
            on_pause=lambda t: ev.emit("status", level="warn",
                                       msg=f"🌡️ Jetson is hot ({t:.0f}°C) – taking a short break to cool down…"),
            on_resume=lambda t: ev.emit("status", msg="❄️ Cooled down – back to learning!"))

        def ema_net(tr):
            return tr.ema.ema if getattr(tr, "ema", None) else tr.model

        def on_train_start(tr):
            st["nb"] = len(tr.train_loader)
            net = ema_net(tr)
            st["groups"] = _param_groups(net)
            st["start_w"] = st["prev_w"] = _snapshot(net)
            n_params = sum(p.numel() for p in net.parameters())
            ev.emit("start", epochs=tr.epochs, nb=st["nb"], device=str(tr.device),
                    params=n_params, batch=tr.batch_size, imgsz=tr.args.imgsz,
                    classes=list(tr.data["names"].values()))
            set_state("running")

        def on_train_epoch_start(tr):
            # nb can change mid-run: Ultralytics halves the batch size after an out-of-memory error
            st.update(in_epoch=True, batch=0, epoch_t0=time.time(), nb=len(tr.train_loader))

        def on_train_batch_end(tr):
            st["batch"] += 1
            guard.check()
            if stop_event.is_set():
                tr.stop = True
            now = time.time()
            if now - st["last_emit"] >= 0.5 or st["batch"] == st["nb"]:
                st["last_emit"] = now
                loss = _losses(tr.tloss)
                if loss:
                    ev.emit("batch", epoch=tr.epoch + 1, batch=st["batch"], nb=st["nb"], loss=loss)

        def predict_previews(tr):
            if not previews or not Path(tr.last).exists():
                return
            try:
                pm = YOLO(str(tr.last))
                res = pm.predict(previews, imgsz=tr.args.imgsz, conf=0.25, device=device,
                                 verbose=False, max_det=20)
                items = []
                for path, r in zip(previews, res):
                    h, w = r.orig_shape
                    dets = []
                    for b in r.boxes:
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        dets.append([r.names[int(b.cls)], _f(b.conf, 3),
                                     _f(x1 / w, 4), _f(y1 / h, 4), _f(x2 / w, 4), _f(y2 / h, 4)])
                    items.append({"image": Path(path).name, "dets": dets})
                ev.emit("preview", epoch=tr.epoch + 1, items=items)
                del pm
            except Exception as e:  # previews are a nice-to-have; never kill training
                print("preview failed:", e)
            finally:
                if device != "cpu":
                    torch.cuda.empty_cache()

        def metrics_of(tr):
            m = tr.metrics or {}
            return {
                "precision": _f(m.get("metrics/precision(B)")),
                "recall": _f(m.get("metrics/recall(B)")),
                "map50": _f(m.get("metrics/mAP50(B)")),
                "map5095": _f(m.get("metrics/mAP50-95(B)")),
                "val_loss": {"box": _f(m.get("val/box_loss")), "cls": _f(m.get("val/cls_loss")),
                             "dfl": _f(m.get("val/dfl_loss"))},
            }

        def on_fit_epoch_end(tr):
            if not st["in_epoch"]:
                # Ultralytics re-fires this after training while validating best.pt
                final = metrics_of(tr)
                run["final"] = final
                ev.emit("final", **final)
                return
            st["in_epoch"] = False
            st["epochs_done"] = st.get("epochs_done", 0) + 1
            if stop_event.is_set():
                tr.stop = True
            ep = tr.epoch + 1
            dt = time.time() - st["epoch_t0"]
            st["epoch_times"].append(dt)
            recent = st["epoch_times"][-3:]
            eta = (sum(recent) / len(recent)) * max(tr.epochs - ep, 0)

            net = ema_net(tr)
            cur = _snapshot(net)
            change = _change(st["groups"], cur, st["prev_w"])
            change_total = _change(st["groups"], cur, st["start_w"])
            st["prev_w"] = cur

            ev.emit("epoch", epoch=ep, epochs=tr.epochs, seconds=_f(dt, 1), eta=_f(eta, 0),
                    train_loss=_losses(tr.tloss),
                    lr=_f(_lr_of(tr), 7), change=change, change_total=change_total,
                    **metrics_of(tr))
            predict_previews(tr)

        def make_model():
            if cfg["pretrained"]:
                m = YOLO(str(weights_path(cfg["model"] + ".pt")))
            else:
                m = YOLO(cfg["model"] + ".yaml")       # random weights: "start from scratch"
            for name, fn in (("on_train_start", on_train_start), ("on_train_epoch_start", on_train_epoch_start),
                             ("on_train_batch_end", on_train_batch_end), ("on_fit_epoch_end", on_fit_epoch_end)):
                m.add_callback(name, fn)
            return m

        epochs = int(cfg["epochs"])
        batch = int(cfg["batch"])
        args = dict(
            data=cfg["yaml"], epochs=epochs, imgsz=int(cfg["imgsz"]), batch=batch,
            # optimizer must be explicit: with the default 'auto' Ultralytics ignores lr0.
            optimizer=cfg["optimizer"], lr0=float(cfg["lr"]),
            warmup_epochs=max(0.0, min(3.0, epochs / 5)),
            close_mosaic=min(10, epochs // 4),
            patience=0,                                   # 0 = never stop early; kids chose the epochs
            device=device, workers=config.TRAIN_WORKERS, cache=False, amp=True,
            project=str(config.RUNS_DIR), name=run_id, exist_ok=True,
            pretrained=bool(cfg["pretrained"]), plots=False, verbose=True, seed=0, deterministic=False,
        )
        if not cfg.get("augment", True):
            args.update(mosaic=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                        translate=0.0, scale=0.0)

        while True:
            model = make_model()
            if cfg.get("freeze"):
                # lock the pretrained "eyes": only the neck and head learn
                args["freeze"] = _backbone_end(model.model) + 1
            print("train args:", args)
            try:
                model.train(**args)
                break
            except Exception as e:
                # Out of memory before the first epoch finished: halve the batch and start over.
                # (Later in the run we'd lose progress, so then we report the error instead.)
                if not (_is_oom(e) and st.get("epochs_done", 0) == 0 and args["batch"] > 1
                        and device != "cpu" and not stop_event.is_set()):
                    raise
                traceback.print_exc()
                old_b = args["batch"]
                args["batch"] = max(1, old_b // 2)
                run["cfg"]["batch"] = args["batch"]
                ev.emit("status", level="warn",
                        msg=f"🧠 The Jetson ran out of memory with batch {old_b}. "
                            f"Trying again with batch {args['batch']}…")
                del model
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                st.update(in_epoch=False, batch=0, epoch_times=[])

        best = run_dir / "weights" / "best.pt"
        if stop_event.is_set():
            set_state("stopped", best=str(best) if best.exists() else None, ended=time.time())
            ev.emit("stopped", best=str(best) if best.exists() else None)
        else:
            set_state("done", best=str(best) if best.exists() else None, ended=time.time())
            ev.emit("done", best=str(best) if best.exists() else None)
    except Exception as e:
        traceback.print_exc()
        msg = str(e)
        if _is_oom(e):
            msg = ("The Jetson ran out of memory. Try a smaller batch size or picture size, the Nano "
                   "model, and close other programs (browser tabs, the camera) while training.")
        set_state("error", error=msg, ended=time.time())
        ev.emit("error", msg=msg)
    finally:
        log.flush()


# ── manager used by the web server ───────────────────────────────────────

class TrainManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._stop_event = None
        self._run_id: str | None = None
        self._event_cache: dict[str, tuple[int, list]] = {}
        self._ev_lock = threading.Lock()     # two tabs polling the same run must not both append a chunk
        self._reserved = False               # set while the GPU is being freed for a run about to start

    @staticmethod
    def mark_interrupted():
        """Runs left 'running' by a previous server (power cut, Ctrl+C) can never finish: say so.
        Called once when the web server starts, never from the training process itself."""
        for p in config.RUNS_DIR.glob("*/run.json"):
            run = _read_json(p)
            if run.get("state") in ("starting", "running"):
                best = p.parent / "weights" / "best.pt"
                run.update(state="stopped" if best.exists() else "error",
                           error="Interrupted: the Lab was closed while this was training.",
                           best=str(best) if best.exists() else None)
                _write_json(p, run)

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def is_busy(self) -> bool:
        """Training is running or about to start: keep other work off the GPU."""
        return self._reserved or self.is_running()

    def reserve(self):
        with self._lock:
            if self.is_busy():
                raise RuntimeError("A model is already training. Wait for it or press Stop.")
            self._reserved = True

    def unreserve(self):
        self._reserved = False

    @property
    def run_id(self):
        return self._run_id

    def start(self, req: dict) -> dict:
        with self._lock:
            if self.is_running():
                raise RuntimeError("A model is already training. Wait for it or press Stop.")
            cfg = self._validate(req)
            run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
            ds = datasets.build(run_id, cfg["classes"], cfg["per_class"],
                                cfg["use_library"], cfg["use_mine"])
            cfg.update(yaml=ds["yaml"], previews=ds["previews"])
            run_dir = config.RUNS_DIR / run_id
            run_dir.mkdir(parents=True)
            _write_json(run_dir / "run.json", {
                "id": run_id, "name": cfg["name"], "created": time.time(), "state": "starting",
                "cfg": cfg, "classes": ds["classes"], "dataset": ds,
            })
            ctx = mp.get_context("spawn")
            self._stop_event = ctx.Event()
            self._proc = ctx.Process(target=_worker, args=(run_id, cfg, self._stop_event),
                                     name=f"train-{run_id}")
            self._proc.start()
            self._run_id = run_id
            threading.Thread(target=self._watch, args=(self._proc, run_id), daemon=True).start()
            return {"run": run_id, "dataset": ds}

    def _watch(self, proc, run_id):
        proc.join()
        run_json = config.RUNS_DIR / run_id / "run.json"
        run = _read_json(run_json)
        if run.get("state") in ("starting", "running"):
            # The worker died without reporting (killed, segfault, out-of-memory killer)
            stopped = bool(self._stop_event and self._stop_event.is_set())
            if stopped:
                reason = "Training was stopped."
            elif proc.exitcode is not None and proc.exitcode < 0:
                # killed by a signal: on the Jetson that is almost always the out-of-memory killer
                reason = (f"Training process was killed (signal {-proc.exitcode}). This usually means the "
                          "computer ran out of memory: try a smaller batch size or picture size.")
            else:
                reason = f"Training process crashed (exit code {proc.exitcode}). See the training log."
            run.update(state="stopped" if stopped else "error", error=reason, ended=time.time())
            _write_json(run_json, run)
            _Events(run_json.parent / "events.jsonl").emit("stopped" if stopped else "error", msg=reason)

    def stop(self, force: bool = False):
        if not self.is_running():
            return False
        self._stop_event.set()
        if force:
            self._proc.terminate()
        else:
            # finish the current batch, validate and save, then exit. Only hard-stop if the run
            # goes completely quiet for 2 minutes: killing it while it rewrites best.pt corrupts the model.
            threading.Thread(target=self._stop_watchdog, args=(self._proc, self._run_id), daemon=True).start()
        return True

    @staticmethod
    def _stop_watchdog(proc, run_id: str, quiet_limit: float = 120.0):
        files = [run_dir(run_id) / "events.jsonl", run_dir(run_id) / "train.log"]
        while proc.is_alive():
            time.sleep(5)
            last = max((f.stat().st_mtime for f in files if f.exists()), default=time.time())
            if time.time() - last > quiet_limit and proc.is_alive():
                proc.terminate()
                return

    @staticmethod
    def _validate(req: dict) -> dict:
        model = req.get("model", "yolov8n")
        if model not in config.MODEL_IDS:
            raise ValueError("Unknown model")
        opt = req.get("optimizer", "SGD")
        if opt not in ("SGD", "AdamW"):
            raise ValueError("Unknown optimizer")

        def num(key, default, lo, hi, cast=int):
            try:
                v = cast(req.get(key, default))
            except (TypeError, ValueError):
                raise ValueError(f"Bad value for {key}")
            return max(lo, min(hi, v))

        imgsz = num("imgsz", 320, 160, 640)
        return {
            "name": str(req.get("name") or "My model")[:60],
            "classes": [str(c).strip().lower() for c in req.get("classes", []) if str(c).strip()],
            "per_class": num("per_class", 100, 2, 2000),
            "use_library": bool(req.get("use_library", True)),
            "use_mine": bool(req.get("use_mine", True)),
            "model": model,
            "pretrained": bool(req.get("pretrained", True)),
            "epochs": num("epochs", 20, 1, 300),
            "batch": num("batch", 8, 1, 64),
            "imgsz": imgsz - imgsz % 32,
            "lr": num("lr", 0.01, 1e-5, 0.5, float),
            "optimizer": opt,
            "augment": bool(req.get("augment", True)),
            "freeze": bool(req.get("freeze", False)),
        }

    # ── reading progress ─────────────────────────────────────────────────
    def events(self, run_id: str, since: int = 0) -> list:
        path = run_dir(run_id) / "events.jsonl"
        if not path.exists():
            return []
        with self._ev_lock:
            return self._events_locked(run_id, path, since)

    def _events_locked(self, run_id: str, path: Path, since: int) -> list:
        offset, cached = self._event_cache.get(run_id, (0, []))
        size = path.stat().st_size
        if size < offset:
            offset, cached = 0, []
        if size > offset:
            with open(path, "rb") as f:
                f.seek(offset)
                chunk = f.read()
            # a trailing partial line is left for the next call
            last_nl = chunk.rfind(b"\n")
            if last_nl >= 0:
                for line in chunk[:last_nl].decode("utf-8", "replace").splitlines():
                    try:
                        cached.append(json.loads(line))
                    except ValueError:
                        pass
                offset += last_nl + 1
            self._event_cache[run_id] = (offset, cached)
        return cached[since:]

    @staticmethod
    def runs() -> list[dict]:
        out = []
        for p in sorted(config.RUNS_DIR.glob("*/run.json"), reverse=True):
            r = _read_json(p)
            if r:
                item = {k: r.get(k) for k in ("id", "name", "created", "ended", "state",
                                              "classes", "final", "error", "cfg")}
                # recomputed, not read from run.json: data/ may have been copied to another
                # machine (laptop -> Jetson) where the old absolute path doesn't exist
                best = p.parent / "weights" / "best.pt"
                item["best"] = str(best) if best.exists() else None
                out.append(item)
        return out

    @staticmethod
    def run(run_id: str) -> dict:
        return _read_json(run_dir(run_id) / "run.json")

    def delete(self, run_id: str):
        if self.is_running() and run_id == self._run_id:
            raise RuntimeError("Can't delete a model while it is training.")
        import shutil
        for d in (run_dir(run_id), build_dir(run_id)):
            d = d.resolve()
            if d.exists() and d.parent in (config.RUNS_DIR.resolve(), config.BUILDS_DIR.resolve()):
                shutil.rmtree(d)
        self._event_cache.pop(run_id, None)


manager = TrainManager()
