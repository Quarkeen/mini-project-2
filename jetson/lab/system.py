"""
Lightweight system monitor: CPU, RAM, temperatures, GPU load.

On the Jetson everything comes from sysfs (no tegrastats subprocess), so
polling it from the UI every few seconds costs next to nothing.
"""
from __future__ import annotations

import glob
import shutil
import subprocess
import time
from pathlib import Path

import psutil

from . import config

_THERMAL_ZONES = []
for _z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
    try:
        _THERMAL_ZONES.append((Path(_z, "type").read_text().strip(), Path(_z, "temp")))
    except OSError:
        pass

# Jetson GPU load file (path differs across JetPack versions)
_GPU_LOAD = next(iter(glob.glob("/sys/devices/platform/bus@0/*.gpu/load")
                      + glob.glob("/sys/devices/platform/*.gpu/load")
                      + glob.glob("/sys/devices/gpu.0/load")), None)

_power_mode = None
_nv_cache = {"t": 0.0, "v": None}


def temperatures() -> dict[str, float]:
    temps = {}
    for name, path in _THERMAL_ZONES:
        try:
            v = int(path.read_text()) / 1000.0
        except Exception:
            # some Jetson sensors error out or return nothing when read (OSError, ValueError,
            # even TypeError from Python's reader): skip that sensor, never fail the caller
            continue
        if -40 < v < 150:
            temps[name] = round(v, 1)
    return temps


def hottest() -> float | None:
    # Orin zones: cpu-thermal, gpu-thermal, soc*-thermal, tj-thermal (junction = max)
    t = temperatures()
    return max(t.values()) if t else None


def jetson_power_mode() -> str | None:
    global _power_mode
    if _power_mode is None and config.IS_JETSON and shutil.which("nvpmodel"):
        try:
            out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=3).stdout
            _power_mode = out.strip().splitlines()[0].replace("NV Power Mode:", "").strip()
        except Exception:
            _power_mode = "unknown"
    return _power_mode


def _nvidia_smi() -> dict | None:
    """Desktop GPU stats, cached for 5 s so polling stays cheap."""
    now = time.time()
    if now - _nv_cache["t"] < 5:
        return _nv_cache["v"]
    _nv_cache["t"] = now
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
        util, temp, used, total = (float(x) for x in out.strip().splitlines()[0].split(","))
        _nv_cache["v"] = {"load": util, "temp": temp, "mem_used_mb": used, "mem_total_mb": total}
    except Exception:
        _nv_cache["v"] = None
    return _nv_cache["v"]


def stats() -> dict:
    vm = psutil.virtual_memory()
    s = {
        "jetson": config.IS_JETSON,
        "cpu": psutil.cpu_percent(interval=None),
        "ram_used_gb": round((vm.total - vm.available) / 1e9, 2),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "temps": temperatures(),
        "hottest": hottest(),
        "thermal_limit": config.THERMAL_PAUSE_C if config.THERMAL_GUARD else None,
    }
    if config.IS_JETSON:
        if _GPU_LOAD:
            try:
                s["gpu"] = {"load": int(Path(_GPU_LOAD).read_text()) / 10.0}
            except (OSError, ValueError):
                pass
        s["power_mode"] = jetson_power_mode()
    else:
        nv = _nvidia_smi()
        if nv:
            s["gpu"] = nv
    return s


class ThermalGuard:
    """Called between training batches; sleeps while the board is too hot."""

    def __init__(self, on_pause=None, on_resume=None, should_abort=None):
        self.enabled = config.THERMAL_GUARD and bool(_THERMAL_ZONES)
        self.on_pause, self.on_resume = on_pause, on_resume
        self.should_abort = should_abort or (lambda: False)
        self._last_check = 0.0

    def check(self):
        """Never raises: a failing temperature read must not stop a training run."""
        try:
            self._check()
        except Exception:
            self.enabled = False      # give up on thermal pausing rather than risk the run

    def _check(self):
        if not self.enabled or time.time() - self._last_check < 2.0:
            return
        self._last_check = time.time()
        t = hottest()
        if t is None or t < config.THERMAL_PAUSE_C:
            return
        if self.on_pause:
            self.on_pause(t)
        while t is not None and t > config.THERMAL_PAUSE_C - config.THERMAL_HYSTERESIS_C:
            if self.should_abort():          # Stop pressed while cooling down
                return
            time.sleep(3)
            t = hottest()
        if self.on_resume:
            self.on_resume(t)
