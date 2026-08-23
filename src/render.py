import cv2
import numpy as np
import pandas as pd
from src.config import EntranceConfig, InteriorConfig
from src.geometry import point_in_polygon
from src.identity import Identity


def draw_polygon(img: np.ndarray, points: list[list[float]] | np.ndarray, color: tuple[int, int, int], thickness: int = 2, fill_alpha: float = 0.0):
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    if fill_alpha > 0.0:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, fill_alpha, img, 1.0 - fill_alpha, 0, img)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_text_with_background(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.5,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    thickness=1,
    padding=3,
):
    (tw, th), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    x, y = org
    x1 = max(0, x - padding)
    y1 = max(0, y - th - padding)
    x2 = min(img.shape[1], x + tw + padding)
    y2 = min(img.shape[0], y + baseline + padding)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(img, text, (x, y), font_face, font_scale, text_color, thickness, cv2.LINE_AA)


def render_entrance_frame(
    frame: np.ndarray,
    frame_idx: int,
    t_s: float,
    identities_map: dict[int, Identity],
    task1_df: pd.DataFrame,
    staff_ids: set[int],
    interactions: list[dict],
    cfg: EntranceConfig,
) -> np.ndarray:
    out = frame.copy()

    # 1. Draw Zones and Zone Labels
    draw_polygon(out, cfg.task1.hallway.points, color=(255, 255, 0), thickness=2, fill_alpha=0.08)  # Cyan
    draw_polygon(out, cfg.task1.storefront_zone.points, color=(0, 255, 255), thickness=1, fill_alpha=0.05)  # Yellow
    draw_polygon(out, cfg.staff.staff_zone.points, color=(255, 100, 0), thickness=2, fill_alpha=0.12)  # Blue

    # Zone tags
    draw_text_with_background(out, "HALLWAY CORRIDOR", (420, 200), font_scale=0.42, text_color=(255, 255, 255), bg_color=(180, 180, 0))
    sz_p = cfg.staff.staff_zone.points[0]
    draw_text_with_background(out, "CASHIER / STAFF ZONE", (int(sz_p[0]) + 10, int(sz_p[1]) + 20), font_scale=0.42, text_color=(255, 255, 255), bg_color=(255, 100, 0))

    # Entrance line
    p1 = tuple(int(x) for x in cfg.task1.entrance_line.p1)
    p2 = tuple(int(x) for x in cfg.task1.entrance_line.p2)
    cv2.line(out, p1, p2, (255, 0, 255), 3, cv2.LINE_AA)
    draw_text_with_background(out, "ENTRANCE THRESHOLD", ((p1[0] + p2[0]) // 2 - 80, (p1[1] + p2[1]) // 2 - 10),
                              font_scale=0.45, text_color=(255, 255, 255), bg_color=(200, 0, 200))

    # 2. Draw Active Interactions
    active_interactions = [
        inter for inter in interactions
        if inter["start_t"] <= t_s <= inter["end_t"]
    ]
    for inter in active_interactions:
        s_id = inter["staff_id"]
        c_id = inter["customer_id"]
        if s_id in identities_map and c_id in identities_map:
            s_ident = identities_map[s_id]
            c_ident = identities_map[c_id]
            if frame_idx in s_ident.frames and frame_idx in c_ident.frames:
                s_idx = np.where(s_ident.frames == frame_idx)[0][0]
                c_idx = np.where(c_ident.frames == frame_idx)[0][0]
                s_pt = tuple(int(x) for x in s_ident.foot_xy[s_idx])
                c_pt = tuple(int(x) for x in c_ident.foot_xy[c_idx])
                cv2.line(out, s_pt, c_pt, (0, 165, 255), 2, cv2.LINE_AA)
                mid = ((s_pt[0] + c_pt[0]) // 2, (s_pt[1] + c_pt[1]) // 2)
                draw_text_with_background(out, f"INTERACTING ({inter['duration_s']:.1f}s)", mid,
                                          font_scale=0.45, text_color=(255, 255, 255), bg_color=(0, 100, 255))

    # 3. Draw Active Identities in this Frame
    t1_dict = {row["identity_id"]: row for _, row in task1_df.iterrows()} if not task1_df.empty else {}
    active_count = 0

    for ident_id, ident in identities_map.items():
        if frame_idx not in ident.frames:
            continue
        active_count += 1
        idx = np.where(ident.frames == frame_idx)[0][0]
        foot = ident.foot_xy[idx]
        h = ident.heights[idx] if len(ident.heights) > idx else 100.0

        is_staff = ident_id in staff_ids
        t1_info = t1_dict.get(ident_id, None)
        is_interested = t1_info["interested"] if t1_info is not None else False
        is_entered = t1_info["entered"] if t1_info is not None else False

        # Color coding
        if is_staff:
            box_color = (255, 100, 0)  # Blue
            label = f"STAFF #{ident_id}"
        elif is_entered:
            box_color = (0, 255, 0)  # Green
            label = f"ENTERED #{ident_id}"
        elif is_interested:
            box_color = (0, 200, 255)  # Orange/Yellow
            label = f"INTERESTED #{ident_id}"
        else:
            box_color = (180, 180, 180)  # Gray
            label = f"ID #{ident_id}"

        # Draw foot point & bounding indicator
        fx, fy = int(foot[0]), int(foot[1])
        cv2.circle(out, (fx, fy), 4, box_color, -1)
        draw_text_with_background(out, label, (fx - 30, max(20, fy - int(h) - 5)),
                                  font_scale=0.45, text_color=(255, 255, 255), bg_color=box_color)

        # Draw facing direction arrow if available
        fn = ident.facing_normal[idx]
        if fn is not None and not np.isnan(fn).any():
            arrow_end = (int(fx + fn[0] * 30), int(fy + fn[1] * 30))
            cv2.arrowedLine(out, (fx, fy), arrow_end, (0, 255, 255), 2, tipLength=0.3)

    # 4. Real-time Running Cumulative HUD Overlay
    if not task1_df.empty:
        seen_mask = task1_df["first_t_s"] <= t_s
        cum_interested = int(task1_df[seen_mask]["interested"].sum())
        cum_entered = int(task1_df[seen_mask]["entered"].sum())
        cum_passed = max(0, cum_interested - cum_entered)
        cum_pedestrians = int(seen_mask.sum())
    else:
        cum_interested = 0
        cum_entered = 0
        cum_passed = 0
        cum_pedestrians = 0

    hud_y = 30
    draw_text_with_background(out, f"ENTRANCE ANALYTICS | Time: {t_s:05.1f}s (Frame {frame_idx}) | In Frame: {active_count}",
                              (20, hud_y), font_scale=0.55, text_color=(255, 255, 255), bg_color=(20, 20, 20), padding=5)
    hud_y += 28
    draw_text_with_background(out, f"Cumulative: Pedestrians: {cum_pedestrians} | Interested: {cum_interested} | Entered: {cum_entered} | Passed By: {cum_passed}",
                              (20, hud_y), font_scale=0.48, text_color=(0, 255, 200), bg_color=(20, 20, 20), padding=4)
    hud_y += 24
    draw_text_with_background(out, f"Staff Detected: {len(staff_ids)} | Active Staff-Customer Interactions: {len(active_interactions)}",
                              (20, hud_y), font_scale=0.45, text_color=(255, 200, 100), bg_color=(20, 20, 20), padding=3)

    return out


SHELF_COLORS = {
    "shelf-a": (255, 105, 180),  # pink
    "shelf-b": (180, 105, 255),  # violet
    "shelf-c": (0, 165, 255),    # orange
    "shelf-d": (255, 255, 0),    # cyan
}


def render_interior_frame(
    frame: np.ndarray,
    frame_idx: int,
    t_s: float,
    identities_map: dict[int, Identity],
    shelf_events_df: pd.DataFrame,
    cfg: InteriorConfig,
) -> np.ndarray:
    """Task 2 overlay: exact annotation rects per shelf, live person->shelf
    association with current interaction duration, and cumulative per-shelf
    interest-event counts (brief §Task 2 visualization requirements)."""
    out = frame.copy()
    shelves = list(cfg.task2.shelves)

    # 1. Shelf rects (exact VIA annotation) + cumulative event count per shelf
    for shelf in shelves:
        col = SHELF_COLORS.get(shelf.name, (0, 255, 0))
        x, y, w, h = [int(v) for v in shelf.rect]
        cum = int(((shelf_events_df["shelf_name"] == shelf.name) & (shelf_events_df["start_t"] <= t_s)).sum()) \
            if not shelf_events_df.empty else 0
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2, cv2.LINE_AA)
        draw_text_with_background(out, f"{shelf.name.upper()}: {cum}",
                                  (x, max(16, y - 6)), font_scale=0.5,
                                  text_color=(255, 255, 255), bg_color=col)

    # 2. Active interactions at this timestamp: association person <-> shelf
    active = [
        row for _, row in shelf_events_df.iterrows()
        if row["start_t"] <= t_s <= row["end_t"]
    ] if not shelf_events_df.empty else []
    active_by_id = {int(r["identity_id"]): r for r in active}
    shelf_by_name = {s.name: s for s in shelves}

    # 3. People
    active_count = 0
    for ident_id, ident in identities_map.items():
        if frame_idx not in ident.frames:
            continue
        active_count += 1
        idx = np.where(ident.frames == frame_idx)[0][0]
        foot = ident.foot_xy[idx]
        h = ident.heights[idx] if len(ident.heights) > idx else 100.0
        fx, fy = int(foot[0]), int(foot[1])

        eng = active_by_id.get(ident_id)
        if eng is not None:
            shelf = shelf_by_name.get(eng["shelf_name"])
            col = SHELF_COLORS.get(eng["shelf_name"], (0, 255, 0))
            elapsed = t_s - float(eng["start_t"])
            label = f"ID #{ident_id} -> {eng['shelf_name']} ({elapsed:.1f}s)"
            if shelf is not None:
                cx, cy = shelf.centroid
                cv2.line(out, (fx, fy), (int(cx), int(cy)), col, 2, cv2.LINE_AA)
                cv2.rectangle(out, (int(shelf.rect[0]), int(shelf.rect[1])),
                              (int(shelf.rect[0] + shelf.rect[2]), int(shelf.rect[1] + shelf.rect[3])),
                              col, 3, cv2.LINE_AA)
        else:
            col = (200, 200, 200)
            label = f"ID #{ident_id}"

        cv2.circle(out, (fx, fy), 4, col, -1)
        draw_text_with_background(out, label, (fx - 30, max(20, fy - int(h) - 5)),
                                  font_scale=0.45, text_color=(255, 255, 255), bg_color=col)

        fn = ident.facing_normal[idx]
        if fn is not None and not np.isnan(fn).any():
            arrow_end = (int(fx + fn[0] * 30), int(fy + fn[1] * 30))
            cv2.arrowedLine(out, (fx, fy), arrow_end, (0, 255, 255), 2, tipLength=0.3)

    # 4. HUD: time + per-shelf cumulative interest-event counts
    total_cum = int((shelf_events_df["start_t"] <= t_s).sum()) if not shelf_events_df.empty else 0
    per_shelf = " | ".join(
        f"{s.name}: {int(((shelf_events_df['shelf_name'] == s.name) & (shelf_events_df['start_t'] <= t_s)).sum()) if not shelf_events_df.empty else 0}"
        for s in shelves
    )
    hud_y = 30
    draw_text_with_background(out, f"INTERIOR ANALYTICS | Time: {t_s:05.1f}s | In Frame: {active_count} | Active interactions: {len(active)}",
                              (20, hud_y), font_scale=0.55, text_color=(255, 255, 255), bg_color=(20, 20, 20), padding=5)
    hud_y += 28
    draw_text_with_background(out, f"Interest events (cumulative) -> {per_shelf} | TOTAL: {total_cum}",
                              (20, hud_y), font_scale=0.48, text_color=(0, 255, 200), bg_color=(20, 20, 20), padding=4)

    return out
