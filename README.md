# AI Vision Lab

**Tools for AI Education on NVIDIA Jetson Orin Nano** · Seventh-semester mini project

AI Vision Lab is a browser-based tool for learning object detection. Students choose
pictures, draw or correct bounding boxes, train a YOLOv8 model, and test it on new
images. Live charts and prediction replay show how the model changes during training.

## Run on a Jetson Orin Nano

These instructions target the project's **Jetson Orin Nano (8 GB)** with
**JetPack 6.1, Ubuntu 22.04, and Python 3.10**. The original Jetson Nano uses a
different software stack and is not covered by this setup. You need internet
access and `sudo` for the initial install. The IMX219 CSI camera is optional;
you can train models and test uploaded photos without it.

### 1. Put the project on the Jetson

If the repository is already on the Jetson, open a terminal in its root directory
(the one containing `jetson/`) and continue to step 2. To copy the app from a
laptop instead, run `hostname -I` on the Jetson to find its IP address. Then
run this **on the laptop, from the repository root**, replacing the example SSH
username and IP:

```bash
JETSON_SSH='your-user@192.168.1.50'
ssh "$JETSON_SSH" 'mkdir -p "$HOME/ai-vision-lab"'
rsync -av jetson/ "${JETSON_SSH}:~/ai-vision-lab/jetson/"
```

### 2. Install the Jetson dependencies once

Run these commands **on the Jetson**, adjusting the path if you cloned the
repository elsewhere:

```bash
cd ~/ai-vision-lab
bash jetson/setup_jetson.sh
```

The script creates `.venv` on the Jetson, installs CUDA-enabled PyTorch for
JetPack, and makes the system OpenCV available to the app. Check that it prints
**`CUDA: True`**. Use this setup script instead of the laptop install commands,
and create the environment on the Jetson rather than copying the laptop's
`.venv`.

### 3. Prepare the picture library

A fresh checkout does not include the picture library or saved experiments.
To download the starter COCO pictures, run this once **on the Jetson**:

```bash
.venv/bin/python jetson/scripts/build_library.py
```

If you already have the project library on your laptop, you can copy it instead.
From the laptop's repository root, with `JETSON_SSH` set as above:

```bash
rsync -avR data/library "${JETSON_SSH}:~/ai-vision-lab/"
```

The existing library includes the project's door and elevator images. To bring
over existing training history and checkpoints, copy `data/runs/` and
`data/builds/` together. Copy `data/models/` too if you need locally imported
models.

### 4. Start the app

From the project root **on the Jetson**:

```bash
.venv/bin/python jetson/run.py --host 0.0.0.0 --port 8888
```

- **On the Jetson:** open [http://localhost:8888](http://localhost:8888).
- **From a laptop or tablet on the same local network:** open
  `http://JETSON_IP:8888`, replacing `JETSON_IP` with the address from
  `hostname -I`.
- Keep the terminal open while using the app. Press `Ctrl+C` to stop it.
  To allow access only from the Jetson itself, use `--host 127.0.0.1`.

For a first training run on the Jetson, choose **pretrained YOLOv8n**,
**320 px**, and **batch size 8**. If memory runs low, reduce the batch size.
To test with the board's ribbon camera, choose **Jetson camera** in the **Test**
tab. **Webcam on this browser** uses the camera of the device displaying the
page.

## Run on a Linux laptop

From the project root, with `uv` installed:

```bash
uv venv --python 3.10 .venv
# For an NVIDIA GPU with a CUDA 12.8-compatible driver:
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r jetson/requirements.txt
.venv/bin/python jetson/scripts/build_library.py  # once, if the library is missing
.venv/bin/python jetson/run.py --host 127.0.0.1 --port 8888
```

For CPU-only use, replace the PyTorch index URL with
`https://download.pytorch.org/whl/cpu`. Open
[http://localhost:8888](http://localhost:8888) after startup.

## Try a lesson

1. **Pictures:** Browse labelled examples or upload a photo. Draw a tight box
   around each object, or correct the AI helper's suggestions. Mark the
   annotation as reviewed so it can be used for training.
2. **Train:** Pick the object classes, dataset size, and YOLOv8 Nano, Small,
   or Medium model. Adjust settings such as epochs, learning rate, batch size,
   image resolution, or backbone freezing. Watch loss, precision, recall,
   mAP, and weight changes. Replay predictions on the same validation pictures
   across epochs.
3. **Test:** Choose a saved checkpoint and try a new photo. Compare the
   predicted boxes, labels, and confidence scores with what you see.

Training runs in a separate process. On the Jetson, the app limits camera
streaming, releases cached inference models before training, and can pause
training between batches if the board gets too hot.

## Repository layout

| Path | Purpose |
|---|---|
| `jetson/run.py` | Starts the web app. |
| `jetson/setup_jetson.sh` | Installs dependencies on the Jetson. |
| `jetson/lab/` | Flask backend, dataset builder, training, inference, and camera support. |
| `jetson/lab/static/` | Browser interface and bundled Chart.js; no frontend build step. |
| `jetson/scripts/build_library.py` | Downloads the starter library and imports the local door/elevator dataset if present. |
| `data/` | Local pictures, labels, model weights, and experiment results. Git excludes these large files. |
