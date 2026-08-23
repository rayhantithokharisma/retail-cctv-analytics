import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


class PoseDetector:
    def __init__(self, weights_path: str, device: str = "cpu", imgsz: int = 1280, conf: float = 0.30):
        self.weights_path = weights_path
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.model = YOLO(weights_path)

    def infer(self, frame: np.ndarray) -> Results:
        """Single call: self.model.predict(frame, classes=[0], conf=self.conf,
        imgsz=self.imgsz, device=self.device, verbose=False)[0]"""
        return self.model.predict(
            frame,
            classes=[0],
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )[0]
