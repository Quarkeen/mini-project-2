"""
Start the AI Vision Lab.

    python run.py                 # http://localhost:8888 (and http://<this-ip>:8888 on the Wi-Fi)
    python run.py --port 8000
    python run.py --dev           # Flask debug server (auto-reload), for development only
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab import config  # noqa: E402


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description="AI Vision Lab")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)    # show the banner immediately under systemd / nohup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.chdir(config.MODELS_DIR)   # stray Ultralytics downloads land in data/models, not wherever you ran this

    from lab.app import create_app
    app = create_app()

    print("\n  AI Vision Lab is running")
    print(f"    on this computer : http://localhost:{args.port}")
    print(f"    on the Wi-Fi     : http://{lan_ip()}:{args.port}")
    print(f"    car endpoint     : http://{lan_ip()}:{args.port}/detect")
    print(f"    device           : {'Jetson' if config.IS_JETSON else 'PC'}, "
          f"{'GPU' if config.torch_device() != 'cpu' else 'CPU'}\n")

    if args.dev:
        app.run(host=args.host, port=args.port, debug=True, threaded=True)
        return
    try:
        from waitress import serve
    except ImportError:
        app.run(host=args.host, port=args.port, threaded=True)
        return
    # channel_timeout keeps long MJPEG streams and long training polls healthy
    serve(app, host=args.host, port=args.port, threads=args.threads, channel_timeout=120,
          ident="ai-vision-lab")


if __name__ == "__main__":
    main()
