"""End-to-end recognizer test. Place test_recognize.jpg in the project root."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: F401
from config import resolve_gpu, settings, setup_logging

import cv2

from database.embeddings import EmbeddingDB
from detection.detector import FaceDetector
from recognition.recognizer import FaceRecognizer

IMAGE_PATH = "test_recognize.jpg"


def main() -> int:
    setup_logging("INFO")
    if not os.path.exists(IMAGE_PATH):
        print(
            f"Place a photo of an enrolled person at {IMAGE_PATH} (project root) and re-run."
        )
        return 1

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Failed to read {IMAGE_PATH}")
        return 1

    use_gpu = resolve_gpu()
    detector = FaceDetector(
        conf_threshold=settings.detection_confidence, use_gpu=use_gpu
    )
    recognizer = FaceRecognizer(use_gpu=use_gpu)
    db = EmbeddingDB(settings.db_path)
    employees = db.list_all()
    db.close()

    if not employees:
        print(
            "No enrolled employees. Run enrollment.py first:\n"
            "  python enrollment.py --employee-id TEST001 --name 'Test User' --photos ./photos/test_user/"
        )
        return 1
    print(f"Loaded {len(employees)} enrolled employee(s)")

    detections = detector.detect(img)
    print(f"Detected {len(detections)} face(s)")
    if not detections:
        return 1

    matched = 0
    for i, d in enumerate(detections):
        emb = recognizer.embed(img, d.bbox)
        if emb is None:
            print(f"  face {i}: failed to embed")
            continue
        match = recognizer.match(emb, employees, settings.recognition_threshold)
        if match:
            print(
                f"  face {i}: MATCH {match.employee_id} ({match.name}) "
                f"score={match.score:.3f}"
            )
            matched += 1
        else:
            # Show best similarity even when below threshold for debugging.
            best_score = -1.0
            best_id = None
            import numpy as np

            e = emb / max(1e-12, float(np.linalg.norm(emb)))
            for rec in employees:
                ref = rec.embedding / max(1e-12, float(np.linalg.norm(rec.embedding)))
                s = float(np.dot(e, ref))
                if s > best_score:
                    best_score = s
                    best_id = rec.employee_id
            print(
                f"  face {i}: no match (best={best_id} score={best_score:.3f}, "
                f"threshold={settings.recognition_threshold:.2f})"
            )

    print(f"Matched {matched}/{len(detections)} face(s)")
    return 0 if matched > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
