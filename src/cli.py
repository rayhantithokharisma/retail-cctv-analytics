from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import typer
from src.config import load_entrance_config, load_interior_config
from src.detection import PoseDetector
from src.identity import build_identities
from src.io_utils import VideoFrameAccessor, open_writer, read_embeddings, read_observations
from src.render import render_entrance_frame, render_interior_frame
from src.stitching import build_tracklets, stitch
from src.tasks.task1_interest import run_task1
from src.tasks.task2_interaction import run_task2_group_dwell
from src.tasks.task2_shelf import shelf_interest_events, shelf_summary
from src.tasks.task3_staff import run_task3, staff_session_summary
from src.tracking import Tracker

app = typer.Typer(help="Retail Behavioral Analytics Pipeline CLI")


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


@app.command()
def run_entrance(
    stride: int = 1,
    limit_s: float | None = None,
    recache: bool = False,
    render: bool = False,
):
    """Executes the complete Entrance pipeline (Perception, Stitching, Task 3 Staff, Task 1 Interest)."""
    cfg = load_entrance_config("configs/entrance.yaml")
    device = pick_device()
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("outputs/debug").mkdir(parents=True, exist_ok=True)

    obs_path = "outputs/debug/entrance_observations.parquet"
    emb_path = "outputs/debug/entrance_embeddings.npy"

    # 1. Perception Pass
    if recache or not (Path(obs_path).exists() and Path(emb_path).exists()):
        print(f"Running Entrance Perception Pass on {device} (stride={stride}, limit_s={limit_s})...")
        detector = PoseDetector("models/yolo11m-pose.pt", device=device, imgsz=1280, conf=0.30)
        tracker = Tracker(detector, "configs/botsort_entrance.yaml")
        cap = cv2.VideoCapture(cfg.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {cfg.path}")

        writer = open_writer(obs_path, embed_path=emb_path)
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total video frames: {total_frames}, FPS: {cfg.fps}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                t_s = frame_idx / cfg.fps
                r = tracker.track(frame)
                embeddings_map = tracker.get_track_embeddings()
                writer.write_frame(
                    video="entrance",
                    frame_idx=frame_idx,
                    t_s=t_s,
                    result=r,
                    embeddings_map=embeddings_map,
                )
                if frame_idx % (stride * 200) == 0:
                    print(f"Entrance frame {frame_idx}/{total_frames} (t={t_s:.1f}s)")
            frame_idx += 1
            if limit_s and (frame_idx / cfg.fps) > limit_s:
                break

        writer.close()
        cap.release()
        print("Entrance Perception Pass complete!")
    else:
        print(f"Using cached Entrance observations from {obs_path}")

    # 2. Offline Stitching & Smoothing
    obs = read_observations(obs_path)
    emb = read_embeddings(emb_path)
    if limit_s:
        obs = obs[obs["t_s"] <= limit_s].copy()

    print("Building tracklets & running Hungarian bipartite stitching...")
    tracklets = build_tracklets(obs, emb)
    stitch_map = stitch(tracklets, cfg.stitch, cfg.scale_map)
    identities = build_identities(obs, stitch_map, cfg)
    print(f"Identified {len(identities)} stitched global identities (from {len(tracklets)} raw tracklets)")

    # 3. Task 3: Staff Detection & Interactions
    print("Running Task 3: Staff Detection & Interaction Analysis...")
    staff_ids, interactions, df_interactions = run_task3(identities, cfg.staff, cfg.task3, fps=cfg.fps)
    print(f"Detected Staff IDs: {sorted(list(staff_ids))}")
    print(f"Found {len(interactions)} staff-customer interaction episodes")

    # 4. Task 1: Store Interest & Conversion
    print("Running Task 1: Store Interest & Entry Classification...")
    task1_df = run_task1(identities, cfg.task1, staff_ids=staff_ids)

    # 5. Export Output CSVs
    task1_df.to_csv("outputs/task1_store_interest.csv", index=False)
    df_interactions.to_csv("outputs/task3_staff_interactions.csv", index=False)

    total_candidates = len(task1_df)
    total_interested = int(task1_df["interested"].sum()) if not task1_df.empty else 0
    total_entered = int(task1_df["entered"].sum()) if not task1_df.empty else 0
    total_passed = max(0, total_interested - total_entered)

    # Brief-required labelled summary tables (final counts must match the videos)
    pd.DataFrame([
        {"metric": "total_interested", "count": total_interested},
        {"metric": "interested_entered", "count": total_entered},
        {"metric": "interested_passed_by", "count": total_passed},
    ]).to_csv("outputs/task1_summary.csv", index=False)
    staff_session_summary(staff_ids, interactions).to_csv("outputs/task3_staff_summary.csv", index=False)

    print("=" * 80)
    print("TASK 1 & TASK 3 SUMMARY METRICS (Entrance)")
    print("=" * 80)
    print(f"Total Hallway Pedestrians (Candidates): {total_candidates}")
    print(f"Total Interested Visitors:             {total_interested}")
    print(f"  - Interested & Entered Store:        {total_entered}")
    print(f"  - Interested & Passed By:            {total_passed}")
    print(f"Staff Identities:                      {sorted(list(staff_ids))}")
    print(f"Total Staff-Customer Interactions:     {len(interactions)}")
    print("=" * 80)
    print("Saved outputs to outputs/task1_store_interest.csv, outputs/task1_summary.csv,")
    print("outputs/task3_staff_interactions.csv and outputs/task3_staff_summary.csv")

    # 6. Render Video
    if render:
        render_entrance_video(limit_s=limit_s)


@app.command()
def run_interior(
    stride: int = 1,
    limit_s: float | None = None,
    recache: bool = False,
    render: bool = False,
):
    """Executes the complete Interior pipeline (Perception, Stitching, Task 2 Interactions & Shelf Attention)."""
    cfg = load_interior_config("configs/interior.yaml")
    device = pick_device()
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("outputs/debug").mkdir(parents=True, exist_ok=True)

    obs_path = "outputs/debug/interior_observations.parquet"
    emb_path = "outputs/debug/interior_embeddings.npy"

    # 1. Perception Pass
    if recache or not (Path(obs_path).exists() and Path(emb_path).exists()):
        print(f"Running Interior Perception Pass on {device} (stride={stride}, limit_s={limit_s})...")
        detector = PoseDetector("models/yolo11m-pose.pt", device=device, imgsz=960, conf=0.30)
        tracker = Tracker(detector, "configs/botsort_interior.yaml")
        cap = cv2.VideoCapture(cfg.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {cfg.path}")

        writer = open_writer(obs_path, embed_path=emb_path)
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total video frames: {total_frames}, FPS: {cfg.fps}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                t_s = frame_idx / cfg.fps
                r = tracker.track(frame)
                embeddings_map = tracker.get_track_embeddings()
                writer.write_frame(
                    video="interior",
                    frame_idx=frame_idx,
                    t_s=t_s,
                    result=r,
                    embeddings_map=embeddings_map,
                )
                if frame_idx % (stride * 200) == 0:
                    print(f"Interior frame {frame_idx}/{total_frames} (t={t_s:.1f}s)")
            frame_idx += 1
            if limit_s and (frame_idx / cfg.fps) > limit_s:
                break

        writer.close()
        cap.release()
        print("Interior Perception Pass complete!")
    else:
        print(f"Using cached Interior observations from {obs_path}")

    # 2. Offline Stitching & Smoothing
    obs = read_observations(obs_path)
    emb = read_embeddings(emb_path)
    if limit_s:
        obs = obs[obs["t_s"] <= limit_s].copy()

    print("Building tracklets & running Hungarian bipartite stitching...")
    tracklets = build_tracklets(obs, emb)
    stitch_map = stitch(tracklets, cfg.stitch, cfg.scale_map)
    identities = build_identities(obs, stitch_map, cfg)
    print(f"Identified {len(identities)} stitched global identities (from {len(tracklets)} raw tracklets)")

    # 3. Task 2: Per-Shelf Customer Interest (+ auxiliary group dwell)
    print("Running Task 2: Per-Shelf Customer Interest Analysis...")
    df_events, df_counts = shelf_interest_events(identities, cfg.task2, fps=cfg.fps)
    df_shelf_summary = shelf_summary(df_events, cfg.task2)
    df_group = run_task2_group_dwell(identities, cfg.task2, fps=cfg.fps)

    # 4. Export Output CSVs
    df_counts.to_csv("outputs/task2_shelf_interest.csv", index=False)
    df_events.to_csv("outputs/task2_shelf_engagement.csv", index=False)
    df_shelf_summary.to_csv("outputs/task2_shelf_summary.csv", index=False)
    df_group.to_csv("outputs/task2_group_dwell.csv", index=False)
    df_events.to_csv("outputs/debug/interior_events.csv", index=False)

    print("=" * 80)
    print("TASK 2 SUMMARY METRICS (Interior)")
    print("=" * 80)
    print("\nShelf Interest Events (Task 2 Official Output):")
    print(df_counts.to_string(index=False))
    print("\nShelf Engagement Breakdown:")
    print(df_shelf_summary.to_string(index=False))
    print(f"\nTotal Group Dwell Episodes (aux):      {len(df_group)}")
    print("=" * 80)
    print("Saved outputs to outputs/task2_*.csv and outputs/debug/interior_events.csv")

    # 5. Render Video
    if render:
        render_interior_video(limit_s=limit_s)


@app.command()
def render_entrance_video(limit_s: float | None = None):
    """Renders annotated video for entrance.mp4 with HUD, zones, and tracking metadata."""
    cfg = load_entrance_config("configs/entrance.yaml")
    obs = read_observations("outputs/debug/entrance_observations.parquet")
    emb = read_embeddings("outputs/debug/entrance_embeddings.npy")
    if limit_s:
        obs = obs[obs["t_s"] <= limit_s].copy()

    tracklets = build_tracklets(obs, emb)
    stitch_map = stitch(tracklets, cfg.stitch, cfg.scale_map)
    identities = build_identities(obs, stitch_map, cfg)
    ident_map = {ident.identity_id: ident for ident in identities}

    staff_ids, interactions, _ = run_task3(identities, cfg.staff, cfg.task3, fps=cfg.fps)
    task1_df = run_task1(identities, cfg.task1, staff_ids=staff_ids)

    cap = cv2.VideoCapture(cfg.path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cfg.fps

    out_path = "outputs/entrance_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"Rendering annotated Entrance video to {out_path}...")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = frame_idx / fps
        rendered = render_entrance_frame(
            frame, frame_idx, t_s, ident_map, task1_df, staff_ids, interactions, cfg
        )
        writer.write(rendered)
        frame_idx += 1
        if limit_s and t_s > limit_s:
            break
        if frame_idx % 300 == 0:
            print(f"Rendered {frame_idx} frames...")

    cap.release()
    writer.release()
    print(f"Render complete: {out_path}")


@app.command()
def render_interior_video(limit_s: float | None = None):
    """Renders annotated video for interior.mp4 with HUD, shelf zones, and group dwells."""
    cfg = load_interior_config("configs/interior.yaml")
    obs = read_observations("outputs/debug/interior_observations.parquet")
    emb = read_embeddings("outputs/debug/interior_embeddings.npy")
    if limit_s:
        obs = obs[obs["t_s"] <= limit_s].copy()

    tracklets = build_tracklets(obs, emb)
    stitch_map = stitch(tracklets, cfg.stitch, cfg.scale_map)
    identities = build_identities(obs, stitch_map, cfg)
    ident_map = {ident.identity_id: ident for ident in identities}

    df_events, _ = shelf_interest_events(identities, cfg.task2, fps=cfg.fps)

    cap = cv2.VideoCapture(cfg.path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cfg.fps

    out_path = "outputs/interior_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"Rendering annotated Interior video to {out_path}...")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = frame_idx / fps
        rendered = render_interior_frame(
            frame, frame_idx, t_s, ident_map, df_events, cfg
        )
        writer.write(rendered)
        frame_idx += 1
        if limit_s and t_s > limit_s:
            break
        if frame_idx % 300 == 0:
            print(f"Rendered {frame_idx} frames...")

    cap.release()
    writer.release()
    print(f"Render complete: {out_path}")


@app.command()
def run_all(stride: int = 1, limit_s: float | None = None, render: bool = False):
    """Runs the entire analytics pipeline on both entrance and interior scenes."""
    print("=" * 80)
    print("RUNNING PIPELINE: ENTRANCE VIDEO")
    print("=" * 80)
    run_entrance(stride=stride, limit_s=limit_s, render=render)

    print("\n" + "=" * 80)
    print("RUNNING PIPELINE: INTERIOR VIDEO")
    print("=" * 80)
    run_interior(stride=stride, limit_s=limit_s, render=render)


if __name__ == "__main__":
    app()
