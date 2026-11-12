"""
Camera attached to the Jetson (IMX219 CSI via GStreamer, or a USB webcam),
streamed to the browser as MJPEG with boxes drawn on it.

One capture thread is shared by all viewers. Detection runs at most
CAMERA_MAX_FPS times per second and the camera is released a few seconds
after the last viewer leaves, so an open browser tab can't keep the GPU busy.
A viewer's stream ends when there is no camera or it stops sending pictures,
so a web-server thread is never held forever.
"""
from __future__ import annotations

import threading
import time

import cv2

from . import config, inference
from .training import manager

PAUSED_FOR_TRAINING = ("The camera is paused while the AI is training: the Jetson needs all its "
                       "memory for learning. Start it again when training has finished.")


def csi_pipeline(w: int, h: int, fps: int = 30) -> str:
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM),width={w},height={h},format=NV12,framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! video/x-raw,width={w},height={h},format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
    )


class JetsonCamera:
    IDLE_SECONDS = 5        # release the camera this long after the last viewer leaves
    RETRY_SECONDS = 5       # after "no camera", don't try to open it again sooner than this
    NO_FRAME_GIVE_UP = 10   # a viewer that gets no new frame for this long is disconnected
    KEEPALIVE_SECONDS = 2   # re-send the last frame so a closed browser tab is noticed

    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._frame = None          # latest annotated JPEG
        self._info = {}
        self._last_viewer = 0.0
        self._retry_after = 0.0
        self._settings = {"model": None, "conf": 0.35}
        self.error = None

    def _open(self):
        w, h = config.CAMERA_SIZE
        if config.IS_JETSON:
            cap = cv2.VideoCapture(csi_pipeline(w, h), cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        return None

    def _run(self):
        cap = self._open()
        if cap is None:
            self.error = "No camera found on this computer."
            self._retry_after = time.time() + self.RETRY_SECONDS
            with self._lock:
                self._thread = None
            return
        self.error = None
        period = 1.0 / config.CAMERA_MAX_FPS
        try:
            while time.time() - self._last_viewer < self.IDLE_SECONDS:
                if manager.is_busy():
                    self.error = PAUSED_FOR_TRAINING
                    break
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                try:
                    res = inference.detect(frame, self._settings["model"], self._settings["conf"],
                                           imgsz=480, annotate=True)
                    self._frame = inference.encode_jpeg(res.pop("annotated_bgr"), 75)
                    self._info = {k: res[k] for k in ("detections", "description", "speed_ms", "device")}
                except Exception as e:
                    self.error = str(e)
                    self._frame = inference.encode_jpeg(frame, 75)
                time.sleep(max(0.0, period - (time.time() - t0)))
        finally:
            cap.release()
            with self._lock:
                self._thread = None
                self._frame = None

    def configure(self, model_path: str, conf: float):
        self._settings.update(model=model_path, conf=conf)

    def touch(self) -> bool:
        """Keep the camera running for this viewer. False if it can't be started right now."""
        if manager.is_busy():
            self.error = PAUSED_FOR_TRAINING
            return False
        self._last_viewer = time.time()
        with self._lock:
            if self._thread is None:
                if time.time() < self._retry_after:
                    return False
                self._thread = threading.Thread(target=self._run, daemon=True, name="jetson-camera")
                self._thread.start()
        return True

    def frames(self):
        """MJPEG generator for one viewer."""
        last, last_sent, last_new = None, 0.0, time.time()
        while True:
            if not self.touch():
                return
            f = self._frame
            now = time.time()
            if f is not None and (f is not last or now - last_sent > self.KEEPALIVE_SECONDS):
                if f is not last:
                    last_new = now
                last, last_sent = f, now
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            elif now - last_new > self.NO_FRAME_GIVE_UP:
                if not self.error:
                    self.error = "The camera stopped sending pictures."
                return
            time.sleep(0.03)

    def stop_for_training(self, timeout: float = 10.0):
        """Release the camera and its GPU model before training starts."""
        self._last_viewer = 0.0          # ends the capture loop
        t = self._thread
        if t is not None:
            t.join(timeout)

    def info(self) -> dict:
        return {"running": self._thread is not None, "error": self.error, **self._info}


camera = JetsonCamera()
