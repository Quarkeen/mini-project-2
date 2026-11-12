# data/ (not in git)

| Folder | Contents | How to get it |
|---|---|---|
| `library/` | Built-in pictures with boxes (COCO + door/elevator) | `jetson/scripts/build_library.py` |
| `mine/` | Pictures uploaded by kids + their boxes (`labels.json`) | the Pictures tab / `import_photos.py` |
| `models/` | Pretrained weights (`yolov8*.pt`, downloaded automatically) and imported models | – |
| `runs/<id>/` | One folder per training run: `run.json`, `events.jsonl` (live charts), `train.log`, `weights/best.pt` | the Train tab |
| `builds/<id>/` | The YOLO dataset assembled for that run (hard links into library/mine) | created by training |
| `car/` | Photos the car sent + what was detected | the car |
| `raw/` | Original project data kept from before the rewrite: own door/elevator photos, test photos, the old merged dataset | – |
| `cache/` | COCO label download | build_library.py |

`models/door_elevator_sept.pt` is the model trained in September (with the fake whole-image
boxes) kept for comparison.
