"""
Import a folder of photos into "My pictures" (they still need boxes checked in the UI).

    python scripts/import_photos.py ../data/raw/own_door_photos --label door
    python scripts/import_photos.py ~/Pictures/lift --label elevator --suggest

--suggest runs the AI helper on every photo so the boxes are pre-drawn and kids only
have to check/fix them. The photos stay "needs checking" until someone saves them in the UI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.imagestore import IMG_EXTS, store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--label", default="", help="what is in these photos (e.g. door)")
    ap.add_argument("--suggest", action="store_true", help="pre-draw boxes with the AI helper")
    ap.add_argument("--source", default="import")
    args = ap.parse_args()

    files = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        sys.exit(f"No pictures found in {args.folder}")
    mine = store("mine")
    names = []
    for p in files:
        names.append(mine.add_image(p, hint=args.label, source=args.source))
        print(f"  + {p.name}")
    print(f"Imported {len(names)} pictures into My pictures (label hint: {args.label or '-'})")

    if args.suggest and args.label:
        from lab import autolabel
        for n in names:
            r = autolabel.suggest(str(mine.image_path(n)), [args.label])
            boxes = [b[:5] for b in r["boxes"]]
            mine.set_boxes(n, boxes, reviewed=False)
            print(f"  {n}: {len(boxes)} suggested box(es) via {', '.join(r['used']) or '-'}")


if __name__ == "__main__":
    main()
