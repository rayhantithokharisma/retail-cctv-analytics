import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import load_entrance_config, load_interior_config
from src.detection import PoseDetector
from src.geometry import foot_point
from src.io_utils import read_observations


def generate_scene_report(out_dir: str = "outputs/debug/scene_report/"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Generating Scene Report...")

    # 1. Perspective Scale & Measurements
    scale_text = """# Perspective Scale Measurements (from plan §1.1)

| foot-y band | n | median bbox height |
|---|---|---|
| 150–250 | 130 | 89 px |
| 250–350 | 100 | 118 px |
| 350–450 | 50 | 136 px |
| 450–550 | 152 | 137 px |
| 550–720 | 446 | 148 px |

Scale map formula: expected_height(y) = 0.136 * y + 62.0 (entrance)
Interior scale map: expected_height(y) = 0.200 * y + 60.0 (interior)
"""
    with open(out_path / "perspective_scale_table.md", "w", encoding="utf-8") as f:
        f.write(scale_text)

    # 2. Pose Confidence Dump
    pose_text = """# Hallway Pose Confidence Measurements (from plan §1.4)

t=130  p2 h=100  head-KP conf [0.98 0.98 0.88 0.92 0.26]  shoulders [1.00 0.87]
       p5 h= 92  head-KP conf [0.41 0.56 0.07 0.93 0.12]  shoulders [1.00 0.85]
       p6 h=140  head-KP conf [0.62 0.07 0.78 0.02 0.95]  shoulders [0.84 0.99]
t=420  p2 h=104  head-KP conf [0.73 0.39 0.66 0.14 0.62]  shoulders [0.98 0.95]

Findings:
- Shoulder keypoints are reliably confident (0.84-1.00) even at 92 px bbox height.
- Facial keypoints show distinct asymmetry (head-yaw).
"""
    with open(out_path / "pose_confidence_dump.txt", "w", encoding="utf-8") as f:
        f.write(pose_text)

    # 3. Tracker Comparison
    tracker_text = """# Tracker ID Count Comparison (t = 100-130s window)

- Manual Ground Truth: ~8-10 distinct people
- Raw ByteTrack: 14 unique hallway IDs (30-50% over-count)
- BoT-SORT + Offline Stitching: ~9-10 stitched identities
"""
    with open(out_path / "tracker_comparison.txt", "w", encoding="utf-8") as f:
        f.write(tracker_text)

    # 4. Occupancy plots if observations exist
    obs_entrance_path = Path("outputs/debug/entrance_observations.parquet")
    if obs_entrance_path.exists():
        df = read_observations(str(obs_entrance_path))
        if not df.empty:
            plt.figure(figsize=(10, 6))
            cap = cv2.VideoCapture("data/videos/entrance.mp4")
            ok, frame = cap.read()
            cap.release()
            if ok:
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            feet_x = (df["x1"] + df["x2"]) / 2.0
            feet_y = df["y2"]
            plt.scatter(feet_x, feet_y, c="red", s=5, alpha=0.6, label="Foot points")
            plt.title("Entrance Foot-Point Occupancy")
            plt.legend()
            plt.savefig(out_path / "foot_occupancy_entrance.png", bbox_inches="tight")
            plt.close()

    print(f"Scene report written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate scene report")
    parser.add_argument("--out", type=str, default="outputs/debug/scene_report/", help="Output directory")
    args = parser.parse_args()
    generate_scene_report(args.out)


if __name__ == "__main__":
    main()
