from functools import lru_cache
from pathlib import Path
from typing import Sequence
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_observation_schema() -> pa.Schema:
    fields = [
        ("video", pa.string()),
        ("frame_idx", pa.int32()),
        ("t_s", pa.float32()),
        ("raw_track_id", pa.int32()),
        ("x1", pa.float32()),
        ("y1", pa.float32()),
        ("x2", pa.float32()),
        ("y2", pa.float32()),
        ("det_conf", pa.float32()),
    ]
    for i in range(17):
        fields.append((f"kp_x_{i:02d}", pa.float32()))
    for i in range(17):
        fields.append((f"kp_y_{i:02d}", pa.float32()))
    for i in range(17):
        fields.append((f"kp_c_{i:02d}", pa.float32()))
    fields.append(("embed_ref", pa.int32()))
    return pa.schema(fields)


class ParquetBatchWriter:
    def __init__(self, parquet_path: str, embed_path: str | None = None, batch_size: int = 5000):
        self.parquet_path = Path(parquet_path)
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        if embed_path is None:
            self.embed_path = self.parquet_path.with_name(
                self.parquet_path.stem.replace("_observations", "_embeddings") + ".npy"
            )
            if self.embed_path == self.parquet_path:
                self.embed_path = self.parquet_path.with_suffix(".npy")
        else:
            self.embed_path = Path(embed_path)
            self.embed_path.parent.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.schema = get_observation_schema()
        self.writer = pq.ParquetWriter(str(self.parquet_path), self.schema)
        self.rows: list[dict] = []
        self.embeddings: list[np.ndarray] = []
        self.is_closed = False

    def write_frame(
        self,
        video: str,
        frame_idx: int,
        t_s: float,
        result,
        embeddings_map: dict[int, np.ndarray] | None = None,
    ):
        if result is None or isinstance(result, list) or result.boxes is None or result.boxes.id is None:
            return

        boxes = result.boxes
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        kpts_data = None
        if result.keypoints is not None and result.keypoints.data is not None:
            kpts_data = result.keypoints.data.cpu().numpy()

        for idx, track_id in enumerate(ids):
            box = xyxy[idx]
            conf = confs[idx]
            embed_ref = -1
            if embeddings_map and track_id in embeddings_map:
                feat = embeddings_map[track_id]
                if feat is not None and len(feat) > 0:
                    self.embeddings.append(np.asarray(feat, dtype=np.float32))
                    embed_ref = len(self.embeddings) - 1

            row = {
                "video": str(video),
                "frame_idx": int(frame_idx),
                "t_s": float(t_s),
                "raw_track_id": int(track_id),
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "det_conf": float(conf),
            }

            if kpts_data is not None and idx < len(kpts_data):
                kp = kpts_data[idx]  # shape (17, 3)
                for k in range(17):
                    row[f"kp_x_{k:02d}"] = float(kp[k, 0])
                    row[f"kp_y_{k:02d}"] = float(kp[k, 1])
                    row[f"kp_c_{k:02d}"] = float(kp[k, 2])
            else:
                for k in range(17):
                    row[f"kp_x_{k:02d}"] = float("nan")
                    row[f"kp_y_{k:02d}"] = float("nan")
                    row[f"kp_c_{k:02d}"] = float(0.0)

            row["embed_ref"] = int(embed_ref)
            self.rows.append(row)

            if len(self.rows) >= self.batch_size:
                self._flush_rows()

    def _flush_rows(self):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        self.writer.write_table(table)
        self.rows.clear()

    def close(self):
        if self.is_closed:
            return
        self._flush_rows()
        self.writer.close()
        if self.embeddings:
            embed_arr = np.vstack(self.embeddings).astype(np.float32)
        else:
            embed_arr = np.empty((0, 512), dtype=np.float32)
        np.save(str(self.embed_path), embed_arr)
        self.is_closed = True


def open_writer(path: str, embed_path: str | None = None, batch_size: int = 5000) -> ParquetBatchWriter:
    return ParquetBatchWriter(path, embed_path=embed_path, batch_size=batch_size)


def read_observations(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_embeddings(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        npy_path = p.with_name(p.stem.replace("_observations", "_embeddings") + ".npy")
        if npy_path.exists():
            return np.load(str(npy_path))
        return np.empty((0, 512), dtype=np.float32)
    return np.load(str(path))


class VideoFrameAccessor:
    def __init__(self, video_path: str, maxsize: int = 128):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.maxsize = maxsize
        self._get_frame_cached = lru_cache(maxsize=maxsize)(self._read_frame_uncached)

    def _read_frame_uncached(self, frame_idx: int) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def get_frame(self, frame_idx: int) -> np.ndarray | None:
        return self._get_frame_cached(frame_idx)

    def close(self):
        if self.cap.isOpened():
            self.cap.release()
