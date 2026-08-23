import numpy as np
from ultralytics.engine.results import Results
from src.detection import PoseDetector


class Tracker:
    def __init__(self, detector: PoseDetector, tracker_yaml: str):
        self.detector = detector
        self.tracker_yaml = tracker_yaml

    def track(self, frame: np.ndarray) -> Results | list:
        """Runs tracking on a frame.
        Returns Results object, or [] if no boxes / no confirmed tracks."""
        res = self.detector.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=self.detector.conf,
            imgsz=self.detector.imgsz,
            device=self.detector.device,
            tracker=self.tracker_yaml,
            verbose=False
        )[0]
        if res.boxes is None or res.boxes.id is None or len(res.boxes.id) == 0:
            return []
        return res

    def get_track_embeddings(self) -> dict[int, np.ndarray]:
        """Extracts current ReID embeddings from the BoT-SORT tracker state."""
        feats = {}
        if hasattr(self.detector.model, "predictor") and hasattr(self.detector.model.predictor, "trackers"):
            trackers = self.detector.model.predictor.trackers
            if trackers and len(trackers) > 0:
                tracker_obj = trackers[0]
                for s in getattr(tracker_obj, "tracked_stracks", []):
                    feat = getattr(s, "curr_feat", None)
                    if feat is None:
                        feat = getattr(s, "smooth_feat", None)
                    if feat is not None:
                        feats[int(s.track_id)] = np.asarray(feat, dtype=np.float32).copy()
        return feats
