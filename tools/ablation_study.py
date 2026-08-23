from pathlib import Path
import pandas as pd
from src.config import load_entrance_config
from src.identity import build_identities
from src.io_utils import read_embeddings, read_observations
from src.stitching import build_tracklets, stitch
from src.tasks.task1_interest import run_task1
from src.tasks.task3_staff import run_task3


def run_ablation_study():
    cfg = load_entrance_config("configs/entrance.yaml")
    obs_path = "outputs/debug/entrance_observations.parquet"
    emb_path = "outputs/debug/entrance_embeddings.npy"

    if not (Path(obs_path).exists() and Path(emb_path).exists()):
        print(f"Observations cache {obs_path} not found. Run entrance inference first.")
        return

    obs = read_observations(obs_path)
    emb = read_embeddings(emb_path)
    tracklets = build_tracklets(obs, emb)

    print("=" * 80)
    print("ABLATION STUDY: Pipeline Configurations Comparison on Entrance Data")
    print("=" * 80)

    # 1. Primary Full Pipeline (BoT-SORT + ReID + Hungarian Stitching + Pose)
    stitch_map_full = stitch(tracklets, cfg.stitch, cfg.scale_map)
    identities_full = build_identities(obs, stitch_map_full, cfg)
    staff_full, inter_full, _ = run_task3(identities_full, cfg.staff, cfg.task3, fps=cfg.fps)
    t1_full = run_task1(identities_full, cfg.task1, staff_ids=staff_full)

    # 2. Ablation 1: No Stitching (Identity == Raw Track)
    stitch_map_none = {t.raw_track_id: t.raw_track_id for t in tracklets}
    identities_nostitch = build_identities(obs, stitch_map_none, cfg)
    staff_nostitch, inter_nostitch, _ = run_task3(identities_nostitch, cfg.staff, cfg.task3, fps=cfg.fps)
    t1_nostitch = run_task1(identities_nostitch, cfg.task1, staff_ids=staff_nostitch)

    # 3. Ablation 2: No ReID in Stitching (appearance weight = 0, motion/scale only)
    cfg_no_reid = cfg.stitch.model_copy(deep=True)
    cfg_no_reid.w_appearance = 0.0
    cfg_no_reid.w_motion = 0.65
    cfg_no_reid.w_scale = 0.35
    stitch_map_no_reid = stitch(tracklets, cfg_no_reid, cfg.scale_map)
    identities_no_reid = build_identities(obs, stitch_map_no_reid, cfg)
    staff_no_reid, inter_no_reid, _ = run_task3(identities_no_reid, cfg.staff, cfg.task3, fps=cfg.fps)
    t1_no_reid = run_task1(identities_no_reid, cfg.task1, staff_ids=staff_no_reid)

    # Summary Table
    rows = [
        {
            "Configuration": "Primary (YOLO11m-Pose + BoT-SORT + ReID + Stitching)",
            "Tracklets / IDs": f"{len(tracklets)} -> {len(identities_full)}",
            "Hallway Candidates": len(t1_full),
            "Interested Total": int(t1_full["interested"].sum()) if not t1_full.empty else 0,
            "Entered": int(t1_full["entered"].sum()) if not t1_full.empty else 0,
            "Staff IDs": len(staff_full),
            "Interactions": len(inter_full),
        },
        {
            "Configuration": "Ablation A: No Stitching (Raw Tracks Only)",
            "Tracklets / IDs": f"{len(tracklets)} -> {len(identities_nostitch)}",
            "Hallway Candidates": len(t1_nostitch),
            "Interested Total": int(t1_nostitch["interested"].sum()) if not t1_nostitch.empty else 0,
            "Entered": int(t1_nostitch["entered"].sum()) if not t1_nostitch.empty else 0,
            "Staff IDs": len(staff_nostitch),
            "Interactions": len(inter_nostitch),
        },
        {
            "Configuration": "Ablation B: Spatial/Scale Stitching Only (No ReID Embeddings)",
            "Tracklets / IDs": f"{len(tracklets)} -> {len(identities_no_reid)}",
            "Hallway Candidates": len(t1_no_reid),
            "Interested Total": int(t1_no_reid["interested"].sum()) if not t1_no_reid.empty else 0,
            "Entered": int(t1_no_reid["entered"].sum()) if not t1_no_reid.empty else 0,
            "Staff IDs": len(staff_no_reid),
            "Interactions": len(inter_no_reid),
        },
    ]

    df_ablation = pd.DataFrame(rows)
    print(df_ablation.to_string(index=False))
    print("=" * 80)
    df_ablation.to_csv("outputs/ablation_study.csv", index=False)
    print("Ablation study exported to outputs/ablation_study.csv")


if __name__ == "__main__":
    run_ablation_study()
