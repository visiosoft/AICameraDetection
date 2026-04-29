"""Greedy IoU-based multi-object tracker for short-lived face tracks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from detection.detector import Detection


BBox = Tuple[int, int, int, int]


@dataclass
class Track:
    track_id: int
    bbox: BBox
    conf: float
    keypoints: Optional[np.ndarray] = None
    hits: int = 1
    misses: int = 0
    stable_recognized: bool = False
    recognized_employee_id: Optional[str] = None
    recognized_name: Optional[str] = None
    recognized_conf: float = 0.0
    last_event_ts: float = 0.0
    embedding_buffer: List[np.ndarray] = field(default_factory=list)


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    def __init__(
        self,
        iou_threshold: float = 0.4,
        max_misses: int = 10,
        stability_frames: int = 3,
    ):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.stability_frames = stability_frames
        self._tracks: List[Track] = []
        self._next_id = 0

    @property
    def tracks(self) -> List[Track]:
        return self._tracks

    def update(self, detections: List[Detection]) -> List[Track]:
        # Build IoU matrix and greedy match by descending IoU.
        pairs: List[Tuple[float, int, int]] = []
        for ti, t in enumerate(self._tracks):
            for di, d in enumerate(detections):
                iou = _iou(t.bbox, d.bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for iou, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            t = self._tracks[ti]
            d = detections[di]
            t.bbox = d.bbox
            t.conf = d.conf
            t.keypoints = d.keypoints
            t.hits += 1
            t.misses = 0

        # Unmatched tracks: increment misses.
        survivors: List[Track] = []
        for ti, t in enumerate(self._tracks):
            if ti not in matched_tracks:
                t.misses += 1
            if t.misses <= self.max_misses:
                survivors.append(t)
        self._tracks = survivors

        # Unmatched detections: spawn new tracks.
        for di, d in enumerate(detections):
            if di in matched_dets:
                continue
            self._tracks.append(
                Track(
                    track_id=self._next_id,
                    bbox=d.bbox,
                    conf=d.conf,
                    keypoints=d.keypoints,
                )
            )
            self._next_id += 1

        return self._tracks

    def needs_recognition(self, track: Track) -> bool:
        return track.hits >= self.stability_frames and not track.stable_recognized
