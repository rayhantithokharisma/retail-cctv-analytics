import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from src.config import load_entrance_config
from src.detection import PoseDetector
from src.geometry import point_in_polygon
from src.identity import build_identities
from src.io_utils import open_writer, read_embeddings, read_observations
from src.stitching import build_tracklets, stitch
from src.tracking import Tracker


def run_window_tracking(
    video_path: str,
    cfg,
    t_start: float,
    t_end: float,
    cache_parquet: str,
    cache_embed: str,
    device: str = "cpu",
    force_recompute: bool = False,
):
    if not force_recompute and Path(cache_parquet).exists() and Path(cache_embed).exists():
        print(f"Using cached observations from {cache_parquet}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cfg.fps
    start_frame = max(0, int((t_start - 2.0) * fps))
    end_frame = int((t_end + 2.0) * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    detector = PoseDetector("models/yolo11m-pose.pt", device=device, imgsz=1280, conf=0.30)
    tracker = Tracker(detector, "configs/botsort_entrance.yaml")

    writer = open_writer(cache_parquet, embed_path=cache_embed)
    frame_idx = start_frame

    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = frame_idx / fps
        r = tracker.track(frame)
        embeddings_map = tracker.get_track_embeddings()
        writer.write_frame("entrance", frame_idx, t_s, r, embeddings_map)
        frame_idx += 1

    writer.close()
    cap.release()


def validate_stitching(fixtures_path: str = "tests/fixtures/hand_counts.json", force_recompute: bool = False):
    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    cfg = load_entrance_config("configs/entrance.yaml")
    hallway_poly = cfg.task1.hallway

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    print("=" * 80)
    print("PHASE 2 VALIDATION GATE: Hand Counts vs. Stitched Identities")
    print("=" * 80)

    for idx, window in enumerate(fixtures.get("entrance", [])):
        t_start = float(window["t_start"])
        t_end = float(window["t_end"])
        hand_n = int(window["n"])

        cache_parquet = f"outputs/debug/val_entrance_w{idx}_obs.parquet"
        cache_embed = f"outputs/debug/val_entrance_w{idx}_emb.npy"

        print(f"\nProcessing Window {idx+1}: [{t_start:.1f}s - {t_end:.1f}s] (Hand count = {hand_n})...")
        run_window_tracking(
            cfg.path,
            cfg,
            t_start,
            t_end,
            cache_parquet,
            cache_embed,
            device=device,
            force_recompute=force_recompute,
        )

        obs = read_observations(cache_parquet)
        emb = read_embeddings(cache_embed)

        # Filter observations to hallway and window
        obs_window = obs[(obs["t_s"] >= t_start) & (obs["t_s"] <= t_end)].copy()
        if obs_window.empty:
            raw_count = 0
            stitched_count = 0
        else:
            # Check raw tracks in hallway
            feet_xy = np.stack(
                [(obs_window["x1"] + obs_window["x2"]) / 2.0, obs_window["y2"]],
                axis=1,
            )
            in_hallway = point_in_polygon(feet_xy, hallway_poly)
            hallway_obs = obs_window[in_hallway]

            raw_hallway_tracks = hallway_obs["raw_track_id"].unique()
            raw_count = len(raw_hallway_tracks)

            # Build tracklets and stitch
            tracklets = build_tracklets(obs, emb)
            stitch_map = stitch(tracklets, cfg.stitch, cfg.scale_map)
            identities = build_identities(obs, stitch_map, cfg)

            # Count stitched identities in hallway during [t_start, t_end]
            stitched_in_hallway = set()
            for ident in identities:
                mask = (ident.t_s >= t_start) & (ident.t_s <= t_end)
                if not np.any(mask):
                    continue
                ident_feet = ident.foot_xy[mask]
                ident_in_h = point_in_polygon(ident_feet, hallway_poly)
                if np.any(ident_in_h):
                    med_height = float(np.median(ident.heights[mask]))
                    duration_s = float(np.sum(ident_in_h) / cfg.fps)
                    if med_height >= cfg.task1.min_bbox_height and duration_s >= 0.5:
                        stitched_in_hallway.add(ident.identity_id)

            stitched_count = len(stitched_in_hallway)

        delta = stitched_count - hand_n
        results.append({
            "window": f"{t_start:.0f}s-{t_end:.0f}s",
            "hand_count": hand_n,
            "raw_tracks": raw_count,
            "stitched_identities": stitched_count,
            "delta": delta,
            "pass": abs(delta) <= 2,
        })

    print("\n" + "-" * 80)
    print(f"{'Window':<15} | {'Hand Count':<12} | {'Raw Tracks':<12} | {'Stitched IDs':<14} | {'Delta':<8} | {'Status'}")
    print("-" * 80)
    all_passed = True
    for r in results:
        status = "PASSED" if r["pass"] else "FAILED"
        if not r["pass"]:
            all_passed = False
        print(f"{r['window']:<15} | {r['hand_count']:<12} | {r['raw_tracks']:<12} | {r['stitched_identities']:<14} | {r['delta']:>+8} | {status}")
    print("-" * 80)

    if all_passed:
        print("\n>>> VALIDATION GATE PASSED (all stitched counts within +/- 1-2 of hand counts) <<<")
    else:
        print("\n>>> VALIDATION GATE FAILED: Stitched counts did not meet +/- 1-2 tolerance! <<<")

    return results, all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate stitching against hand counts")
    parser.add_argument("--recache", action="store_true", help="Force re-running tracker")
    args = parser.parse_args()
    validate_stitching(force_recompute=args.recache)


if __name__ == "__main__":
    main()
