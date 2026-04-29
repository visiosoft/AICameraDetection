"""InsightFace buffalo_s based recognizer."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from database.embeddings import EmployeeRecord

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    employee_id: str
    name: str
    score: float


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return v
    return v / n


class FaceRecognizer:
    def __init__(self, use_gpu: bool = False, det_size: Tuple[int, int] = (640, 640)):
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name="buffalo_s", providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)
        self._use_gpu = use_gpu

        # Cache recognition model for direct landmark-based embedding
        self._rec_model = self.app.models.get('recognition')
        self._norm_crop = None
        if self._rec_model is not None:
            try:
                from insightface.utils.face_align import norm_crop
                self._norm_crop = norm_crop
            except ImportError:
                self._rec_model = None
        logger.info(
            "InsightFace loaded (model=buffalo_s, providers=%s, direct_embed=%s)",
            providers, self._rec_model is not None,
        )

    def embed_from_full_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Pick the highest-scoring face from a full image and return its embedding."""
        if frame is None or frame.size == 0:
            return None
        faces = self.app.get(frame)
        if not faces:
            return None
        best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        emb = getattr(best, "normed_embedding", None)
        if emb is None:
            emb = _l2_normalize(np.asarray(best.embedding, dtype=np.float32))
        return np.asarray(emb, dtype=np.float32)

    def embed(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin: float = 0.2,
    ) -> Optional[np.ndarray]:
        """Embed the face inside bbox, expanding the crop by `margin` on each side."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        mx = int(bw * margin)
        my = int(bh * margin)
        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        return self.embed_from_full_frame(crop)

    def embed_aligned(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Embed face using pre-detected landmarks, bypassing InsightFace's detector."""
        if self._rec_model is None or self._norm_crop is None:
            return None
        if frame is None or frame.size == 0:
            return None
        try:
            aligned = self._norm_crop(frame, keypoints, image_size=112)
            feat = self._rec_model.get_feat(aligned).flatten()
            return _l2_normalize(np.asarray(feat, dtype=np.float32))
        except Exception:
            logger.debug("embed_aligned failed, falling back", exc_info=True)
            return None

    def match(
        self,
        embedding: np.ndarray,
        records: Iterable[EmployeeRecord],
        threshold: float,
    ) -> Optional[MatchResult]:
        if embedding is None:
            return None
        e = _l2_normalize(np.asarray(embedding, dtype=np.float32))
        best: Optional[MatchResult] = None
        best_score = -1.0
        for rec in records:
            ref = _l2_normalize(np.asarray(rec.embedding, dtype=np.float32))
            score = float(np.dot(e, ref))
            if score > best_score:
                best_score = score
                best = MatchResult(
                    employee_id=rec.employee_id,
                    name=rec.name,
                    score=score,
                )
        if best is None:
            return None
        if best.score >= threshold:
            return best
        logger.debug(
            "Face detected but no match (best similarity: %.2f for %s)",
            best.score,
            best.employee_id,
        )
        return None

    def match_fast(
        self,
        embedding: np.ndarray,
        emb_matrix: np.ndarray,
        records: list,
        threshold: float,
    ) -> Optional[MatchResult]:
        """Vectorized matching against a precomputed normalized embedding matrix."""
        if embedding is None or emb_matrix.shape[0] == 0:
            return None
        e = _l2_normalize(np.asarray(embedding, dtype=np.float32))
        scores = emb_matrix @ e
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= threshold:
            rec = records[best_idx]
            return MatchResult(
                employee_id=rec.employee_id,
                name=rec.name,
                score=best_score,
            )
        logger.debug("No match (best similarity: %.2f)", best_score)
        return None
