import csv
import json
from functools import cached_property
from pathlib import Path
from typing import Literal
import matplotlib.path
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import yaml


class ScaleMap(BaseModel):
    a: float
    b: float  # expected_height(y) = a*y + b

    def expected_height(self, y: float) -> float:
        return float(max(1.0, self.a * y + self.b))


class Polygon(BaseModel):
    points: list[tuple[float, float]]  # >=3 points, closed implicitly
    model_config = ConfigDict(ignored_types=(cached_property,))

    @cached_property
    def to_path(self) -> matplotlib.path.Path:
        return matplotlib.path.Path(self.points)


class Line(BaseModel):
    p1: tuple[float, float]
    p2: tuple[float, float]
    store_side_sign: Literal[-1, 1]

    def signed_side(self, q: tuple[float, float] | list[float] | np.ndarray) -> float:
        # cross(Q) = d_x*(Q_y - P1_y) - d_y*(Q_x - P1_x)
        d_x = self.p2[0] - self.p1[0]
        d_y = self.p2[1] - self.p1[1]
        if isinstance(q, np.ndarray) and q.ndim == 2:
            cross = d_x * (q[:, 1] - self.p1[1]) - d_y * (q[:, 0] - self.p1[0])
            return cross * self.store_side_sign
        qx = q[0]
        qy = q[1]
        cross = d_x * (qy - self.p1[1]) - d_y * (qx - self.p1[0])
        return float(cross * self.store_side_sign)


class TrackerConfig(BaseModel):
    tracker_type: Literal["botsort", "bytetrack"]
    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    track_buffer: int
    match_thresh: float
    fuse_score: bool
    gmc_method: str
    with_reid: bool
    proximity_thresh: float
    appearance_thresh: float


class StitchConfig(BaseModel):
    max_gap_frames: int
    motion_gate_bh: float
    scale_gate_frac: float
    direction_gate_deg: float
    cost_max: float
    w_appearance: float
    w_motion: float
    w_scale: float


class EventConfig(BaseModel):
    t_on_s: float
    t_off_s: float


class Task1Config(BaseModel):
    hallway: Polygon
    entrance_line: Line
    storefront_zone: Polygon
    min_bbox_height: float
    w_orientation: float
    w_decel: float
    w_approach: float
    w_dwell: float
    orientation_cos_floor_deg: float  # 75
    interest_threshold: float  # 0.55
    interest_min_duration_s: float  # 0.8
    entered_dwell_s: float
    entered_depth_bh: float
    entered_deadzone_bh: float
    event: EventConfig


class ShelfConfig(BaseModel):
    """A designated shelf. Geometry comes straight from the VIA annotation rect
    (x, y, w, h in pixels @1280x720) — the rect marks the product area of the shelf."""
    name: str
    rect: tuple[float, float, float, float]  # x, y, width, height (VIA annotation)
    min_bbox_height: float | None = None  # per-shelf override
    facing_cos_floor_deg: float | None = None  # per-shelf override
    model_config = ConfigDict(ignored_types=(cached_property,))

    @cached_property
    def face_polygon(self) -> Polygon:
        x, y, w, h = self.rect
        return Polygon(points=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

    @cached_property
    def centroid(self) -> tuple[float, float]:
        x, y, w, h = self.rect
        return (x + w / 2.0, y + h / 2.0)


class Task2Config(BaseModel):
    shelves: list[ShelfConfig] = []
    annotation_csv: str | None = None  # VIA CSV; source of truth for shelf rects
    ignore_zones: list[Polygon] = []  # feet inside are never shelf candidates
    reach_dist_bh: float = 1.2  # max foot-to-shelf-rect distance, in body heights
    facing_cos_floor_deg: float = 80.0  # torso must face the shelf within this angle
    engage_speed_bh_s: float = 0.35  # browsing == slower than this (body heights / s)
    min_bbox_height: float = 40.0  # ignore tiny/far detections
    event: EventConfig


class StaffConfig(BaseModel):
    apron_colours: list[dict]  # name, h:[lo,hi], s:[lo,hi], v:[lo,hi]
    staff_zone: Polygon
    w_apron: float
    w_counter: float
    w_dwell: float
    staff_score_threshold: float
    counter_prior_norm_s: float
    dwell_prior_norm_s: float


class Task3Config(BaseModel):
    proximity_bh: float
    orientation_cos_floor_deg: float
    co_stationary_bh_s: float
    event: EventConfig


class VideoConfig(BaseModel):
    path: str
    fps: float
    width: int
    height: int
    scale_map: ScaleMap
    tracker: TrackerConfig
    stitch: StitchConfig
    smoothing: dict  # median_win, savgol_win, savgol_order, max_interp_gap


class EntranceConfig(VideoConfig):
    task1: Task1Config
    task3: Task3Config
    staff: StaffConfig


class InteriorConfig(VideoConfig):
    task2: Task2Config
    staff: StaffConfig


def load_entrance_config(path: str) -> EntranceConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return EntranceConfig(**data)


def load_via_shelf_rects(csv_path: str) -> dict[str, tuple[float, float, float, float]]:
    """Parse a VIA annotation CSV and return {region_name: (x, y, w, h)} for rect regions."""
    rects: dict[str, tuple[float, float, float, float]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            shape_raw = row.get("region_shape_attributes", "")
            attr_raw = row.get("region_attributes", "")
            if not shape_raw:
                continue
            shape = json.loads(shape_raw)
            attr = json.loads(attr_raw) if attr_raw else {}
            if shape.get("name") != "rect" or "name" not in attr:
                continue
            rects[attr["name"]] = (
                float(shape["x"]),
                float(shape["y"]),
                float(shape["width"]),
                float(shape["height"]),
            )
    return rects


def load_interior_config(path: str) -> InteriorConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = InteriorConfig(**data)

    # Shelf geometry is sourced from the VIA annotation CSV (single source of truth);
    # per-shelf YAML entries (if any) only carry threshold overrides, keyed by name.
    task2 = data.get("task2", {})
    csv_path = task2.get("annotation_csv")
    if csv_path:
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Task2 annotation_csv not found: {csv_path}")
        overrides = {s["name"]: s for s in task2.get("shelves", []) if isinstance(s, dict)}
        shelves = []
        for name, rect in sorted(load_via_shelf_rects(csv_path).items()):
            ov = overrides.get(name, {})
            shelves.append(ShelfConfig(
                name=name,
                rect=rect,
                min_bbox_height=ov.get("min_bbox_height"),
                facing_cos_floor_deg=ov.get("facing_cos_floor_deg"),
            ))
        cfg.task2.shelves = shelves
    return cfg
