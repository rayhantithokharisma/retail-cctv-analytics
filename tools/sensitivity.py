import argparse
from pathlib import Path
import pandas as pd
from src.config import load_entrance_config, load_interior_config
from src.identity import build_identities
from src.io_utils import read_embeddings, read_observations
from src.stitching import build_tracklets, stitch
from src.tasks.task1_interest import run_task1
from src.tasks.task2_shelf import shelf_interest_events
from src.tasks.task3_staff import run_task3


def run_sensitivity_analysis():
    entrance_cfg = load_entrance_config("configs/entrance.yaml")
    interior_cfg = load_interior_config("configs/interior.yaml")

    # 1. Entrance Sensitivity
    ent_obs_path = "outputs/debug/entrance_observations.parquet"
    ent_emb_path = "outputs/debug/entrance_embeddings.npy"

    if Path(ent_obs_path).exists() and Path(ent_emb_path).exists():
        obs = read_observations(ent_obs_path)
        emb = read_embeddings(ent_emb_path)
        tracklets = build_tracklets(obs, emb)
        stitch_map = stitch(tracklets, entrance_cfg.stitch, entrance_cfg.scale_map)
        identities = build_identities(obs, stitch_map, entrance_cfg)
        staff_ids, _, _ = run_task3(identities, entrance_cfg.staff, entrance_cfg.task3, fps=entrance_cfg.fps)

        print("\n" + "=" * 80)
        print("TASK 1 SENSITIVITY TABLE: Interest Threshold vs. t_off_s")
        print("=" * 80)
        print(f"{'Interest Thresh':<18} | {'t_off (s)':<10} | {'Total Interested':<18} | {'Entered':<10} | {'Passed By':<10}")
        print("-" * 80)

        for thresh in [0.45, 0.55, 0.65]:
            for t_off in [2.0, 3.0, 4.0]:
                cfg_copy = entrance_cfg.task1.model_copy(deep=True)
                cfg_copy.interest_threshold = thresh
                cfg_copy.event.t_off_s = t_off

                df1 = run_task1(identities, cfg_copy, staff_ids=staff_ids)
                tot_int = int(df1["interested"].sum()) if not df1.empty else 0
                tot_ent = int(df1["entered"].sum()) if not df1.empty else 0
                tot_pass = max(0, tot_int - tot_ent)
                print(f"{thresh:<18.2f} | {t_off:<10.1f} | {tot_int:<18} | {tot_ent:<10} | {tot_pass:<10}")
        print("-" * 80)

    # 2. Interior Sensitivity
    int_obs_path = "outputs/debug/interior_observations.parquet"
    int_emb_path = "outputs/debug/interior_embeddings.npy"

    if Path(int_obs_path).exists() and Path(int_emb_path).exists():
        obs_int = read_observations(int_obs_path)
        emb_int = read_embeddings(int_emb_path)
        tracklets_int = build_tracklets(obs_int, emb_int)
        stitch_map_int = stitch(tracklets_int, interior_cfg.stitch, interior_cfg.scale_map)
        identities_int = build_identities(obs_int, stitch_map_int, interior_cfg)

        print("\n" + "=" * 80)
        print("TASK 2 SENSITIVITY TABLE: Reach Distance (BH) vs. t_off_s")
        print("=" * 80)
        print(f"{'Reach dist (BH)':<18} | {'t_off (s)':<10} | {'Shelf A':<10} | {'Shelf B':<10} | {'Shelf C':<10} | {'Shelf D':<10} | {'Total Events':<12}")
        print("-" * 80)

        for reach_bh in [1.0, 1.2, 1.5]:
            for t_off in [2.0, 3.0, 4.0]:
                cfg_copy = interior_cfg.task2.model_copy(deep=True)
                cfg_copy.reach_dist_bh = reach_bh
                cfg_copy.event.t_off_s = t_off

                _, df_counts = shelf_interest_events(identities_int, cfg_copy, fps=interior_cfg.fps)
                counts = {row["shelf"]: row["interest_events"] for _, row in df_counts.iterrows()}
                print(f"{reach_bh:<18.2f} | {t_off:<10.1f} | {counts.get('shelf-a', 0):<10} | {counts.get('shelf-b', 0):<10} | {counts.get('shelf-c', 0):<10} | {counts.get('shelf-d', 0):<10} | {counts.get('total', 0):<12}")
        print("-" * 80)


if __name__ == "__main__":
    run_sensitivity_analysis()
