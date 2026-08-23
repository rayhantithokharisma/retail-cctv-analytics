"""Debug tool: draw VIA annotation rects and/or config shelf polygons on interior frames.

Usage:
    python tools/draw_shelves.py --mode via      # ground-truth annotation rects
    python tools/draw_shelves.py --mode config   # current config face/zone polygons
    python tools/draw_shelves.py --mode both
"""
import argparse
import ast
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src.config import load_interior_config

VIA_CSV = "data/annotations/interior_annotation.csv"
OUT_DIR = Path("outputs/debug/shelf_check")

SHELF_COLORS = {
    "shelf-a": (255, 105, 180),  # pink-ish (BGR-ish)
    "shelf-b": (180, 105, 255),
    "shelf-c": (0, 165, 255),    # orange
    "shelf-d": (255, 255, 0),    # cyan
}


def load_via_rects() -> dict[str, tuple[int, int, int, int]]:
    rects = {}
    with open(VIA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shape = json.loads(row["region_shape_attributes"])
            attr = json.loads(row["region_attributes"])
            name = attr["name"]
            rects[name] = (int(shape["x"]), int(shape["y"]), int(shape["width"]), int(shape["height"]))
    return rects


def draw_via(img: np.ndarray) -> np.ndarray:
    rects = load_via_rects()
    for name, (x, y, w, h) in rects.items():
        color = SHELF_COLORS.get(name, (0, 255, 0))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"VIA:{name}", (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def draw_config(img: np.ndarray) -> np.ndarray:
    cfg = load_interior_config("configs/interior.yaml")
    for shelf in cfg.task2.shelves:
        color = SHELF_COLORS.get(shelf.name, (0, 255, 0))
        x, y, w, h = [int(v) for v in shelf.rect]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"CFG:{shelf.name}", (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["via", "config", "both"], default="both")
    ap.add_argument("--frames", nargs="*", default=None, help="frame paths; default: all samples")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = args.frames or sorted(str(p) for p in Path("data/samples/interior").glob("*.jpg"))
    for fp in frames:
        img = cv2.imread(fp)
        if img is None:
            continue
        if args.mode in ("via", "both"):
            img = draw_via(img)
        if args.mode in ("config", "both"):
            img = draw_config(img)
        out = OUT_DIR / f"{Path(fp).stem}_{args.mode}.jpg"
        cv2.imwrite(str(out), img)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
