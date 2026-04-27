"""Test face detection on live RTSP stream.

Run this script to verify:
1. RTSP stream connectivity
2. Face detection is working
3. Visual output with bounding boxes

Press 'q' to quit.
"""
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: F401  (sets OPENCV_FFMPEG_CAPTURE_OPTIONS)
from config import resolve_gpu, settings, setup_logging

import cv2

from detection.detector import FaceDetector


def main() -> int:
    setup_logging()
    print("\n" + "=" * 60)
    print("Face Detection Live Test")
    print("=" * 60)
    print(f"RTSP URL: {settings.rtsp_url}")
    print(f"Detection confidence: {settings.detection_confidence}")
    print(f"Using GPU: {resolve_gpu()}")
    print("=" * 60)
    print("\nPress 'q' to quit\n")

    use_gpu = resolve_gpu()
    detector = FaceDetector(
        conf_threshold=settings.detection_confidence,
        use_gpu=use_gpu,
    )

    print("Connecting to RTSP stream...")
    cap = cv2.VideoCapture(settings.rtsp_url)

    if not cap.isOpened():
        print(f"❌ Failed to connect to {settings.rtsp_url}")
        return 1

    print("✓ Connected to stream!\n")

    frame_count = 0
    detection_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            frame_count += 1

            # Detect faces
            detections = detector.detect(frame)

            if detections:
                detection_count += 1
                print(f"Frame {frame_count}: Detected {len(detections)} face(s)")
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det.bbox
                    conf = det.conf
                    print(f"  Face {i+1}: bbox=({x1},{y1},{x2},{y2}) confidence={conf:.2f}")

            # Draw bounding boxes
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Face {det.conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Add info overlay
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            info_text = f"FPS: {fps:.1f} | Frames: {frame_count} | Detections: {detection_count}"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Face Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print(f"Test Summary:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Frames with detections: {detection_count}")
        print(f"  Detection rate: {100*detection_count/frame_count if frame_count > 0 else 0:.1f}%")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
