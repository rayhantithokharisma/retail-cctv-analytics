"""Frame-by-frame visual verification of Task 2 shelf interest events.

Renders the SAME annotated frames used in the final video at the start/middle/end
of every detected event, so each event can be eyeballed against the footage.

Usage: PYTHONPATH=. python tools/verify_task2.py
"""
from pathlib import Path

import cv2
import pandas as pd

from src.config import load_interior_config
from src.identity import build_identities
from src.io_utils import read_embeddings, read_observations
from src.render import render_interior_frame
from src.stitching import build_tracklets, stitch
from src.tasks.task2_shelf import shelf_interest_events

OUT_DIR = Path("outputs/debug/task2_verify")


def main():
    cfg = load_interior_config("configs/interior.yaml")
    obs = read_observations("outputs/debug/interior_observations.parquet")
    emb = read_embeddings("outputs/debug/interior_embeddings.npy")
    tracklets = build_tracklets(obs, emb)
    stitch_map = stitch(tracklets, cfg.stitch, cfg.scale_map)
    identities = build_identities(obs, stitch_map, cfg)
    ident_map = {i.identity_id: i for i in identities}

    df_events, df_counts = shelf_interest_events(identities, cfg.task2, fps=cfg.fps)
    print(df_counts.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(cfg.path)

    for n, ev in enumerate(df_events.itertuples(index=False), start=1):
        for phase, t in [("start", ev.start_t + 0.5), ("mid", (ev.start_t + ev.end_t) / 2), ("end", ev.end_t - 0.5)]:
            frame_idx = int(round(t * cfg.fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            t_s = frame_idx / cfg.fps
            rendered = render_interior_frame(frame, frame_idx, t_s, ident_map, df_events, cfg)
            out = OUT_DIR / f"ev{n:02d}_id{ev.identity_id}_{ev.shelf_name}_{phase}_{t_s:06.1f}s.jpg"
            cv2.imwrite(str(out), rendered)
        print(f"event {n}: id={ev.identity_id} {ev.shelf_name} {ev.start_t:.1f}-{ev.end_t:.1f}s -> dumped")

    cap.release()
    print(f"\nWrote verification frames to {OUT_DIR}/")


if __name__ == "__main__":
    main()
