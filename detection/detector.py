"""YOLOv8-face detector wrapper."""
from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/"
    "yolov10n-face.pt"
)
DEFAULT_MODEL_PATH = os.path.join("data", "models", "yolov10n-face.pt")


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    conf: float
    keypoints: Optional[np.ndarray] = None  # (5, 2) facial landmarks if available


def ensure_model(path: str = DEFAULT_MODEL_PATH, url: str = DEFAULT_MODEL_URL) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    logger.info("Downloading YOLOv8-face weights from %s", url)
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    logger.info("Saved YOLOv8-face weights to %s", path)
    return path


class FaceDetector:
    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.5,
        use_gpu: bool = False,
    ):
        from ultralytics import YOLO

        path = ensure_model(model_path or DEFAULT_MODEL_PATH)
        self.conf = float(conf_threshold)
        self.device = self._select_device(use_gpu)
        self.model = YOLO(path)
        try:
            self.model.to(self.device)
        except Exception:
            # Some ultralytics versions accept device only at predict time.
            pass
        logger.info("YOLO loaded on %s", self.device)

    @staticmethod
    def _select_device(use_gpu: bool) -> str:
        if not use_gpu:
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []
        results = self.model.predict(
            frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        # Extract facial landmarks if the model provides them
        kps_array = None
        try:
            if r.keypoints is not None:
                kps_array = r.keypoints.xy.cpu().numpy()  # (N, 5, 2)
        except (AttributeError, IndexError):
            pass

        h, w = frame.shape[:2]
        for i, ((x1, y1, x2, y2), c) in enumerate(zip(xyxy, confs)):
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(w - 1, int(round(x2)))
            y2 = min(h - 1, int(round(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            kp = None
            if kps_array is not None and i < len(kps_array):
                kp = kps_array[i].astype(np.float32)  # (5, 2)
                if np.any(kp[:, 0] <= 0) or np.any(kp[:, 1] <= 0):
                    kp = None  # invalid landmarks
            out.append(Detection(bbox=(x1, y1, x2, y2), conf=float(c), keypoints=kp))
        return out
