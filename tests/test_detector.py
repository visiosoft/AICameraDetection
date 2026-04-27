"""Run YOLOv8-face on a static image. Place test_face.jpg in the project root."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: F401
from config import resolve_gpu, settings, setup_logging

import cv2

from detection.detector import FaceDetector

IMAGE_PATH = "test_face.jpg"


def main() -> int:
    setup_logging("INFO")
    if not os.path.exists(IMAGE_PATH):
        print(f"Place an image with faces at {IMAGE_PATH} (project root) and re-run.")
        return 1

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Failed to read {IMAGE_PATH}")
        return 1

    detector = FaceDetector(
        conf_threshold=settings.detection_confidence,
        use_gpu=resolve_gpu(),
    )
    detections = detector.detect(img)
    print(f"Detected {len(detections)} face(s)")
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d.bbox
        print(f"  [{i}] bbox={d.bbox} conf={d.conf:.3f}")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            img,
            f"{d.conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("test_detector", img)
    print("Press any key in the window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
