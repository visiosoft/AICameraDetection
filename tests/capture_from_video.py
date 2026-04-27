"""Capture faces from live video stream for enrollment."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from config import resolve_gpu, settings, setup_logging

import cv2

from detection.detector import FaceDetector


def main() -> int:
    setup_logging()
    
    print("\n" + "=" * 70)
    print("VIDEO FACE CAPTURE TOOL")
    print("=" * 70)
    print(f"RTSP URL: {settings.rtsp_url}")
    print(f"Detection confidence: {settings.detection_confidence}")
    print("=" * 70)
    print("\nInstructions:")
    print("  • Face the camera directly")
    print("  • Press SPACE to capture a face")
    print("  • Capture 3-5 photos from different angles")
    print("  • Press Q to quit")
    print("=" * 70 + "\n")

    employee_id = input("Enter employee ID (e.g., TEST003): ").strip()
    if not employee_id:
        print("Employee ID required!")
        return 1
    
    name = input("Enter employee name: ").strip()
    if not name:
        print("Name required!")
        return 1
    
    # Create output directory
    output_dir = Path("photos") / employee_id.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPhotos will be saved to: {output_dir}")

    # Connect to stream
    print("\nConnecting to video stream...")
    cap = cv2.VideoCapture(settings.rtsp_url)
    
    if not cap.isOpened():
        print(f"❌ Cannot connect to {settings.rtsp_url}")
        return 1
    
    print("✓ Connected")

    # Load detector
    print("Loading face detector...")
    use_gpu = resolve_gpu()
    detector = FaceDetector(conf_threshold=settings.detection_confidence, use_gpu=use_gpu)
    print("✓ Detector loaded\n")

    print("=" * 70)
    print("LIVE VIEW - Press SPACE to capture, Q to quit")
    print("=" * 70 + "\n")

    capture_count = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            frame_count += 1
            
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            # Detect faces
            display_frame = frame.copy()
            detections = detector.detect(frame)
            
            # Draw bounding boxes
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Face {det.conf:.2f}"
                cv2.putText(
                    display_frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            
            # Add instructions overlay
            cv2.putText(
                display_frame,
                f"Detected: {len(detections)} face(s) | Captured: {capture_count} photos",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                "SPACE = Capture | Q = Quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            
            cv2.imshow("Face Capture", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord(' '):  # Space bar
                if len(detections) == 0:
                    print("⚠️  No faces detected. Move closer or adjust lighting.")
                elif len(detections) > 1:
                    print(f"⚠️  Multiple faces detected ({len(detections)}). Please ensure only one person is visible.")
                else:
                    capture_count += 1
                    filename = output_dir / f"{capture_count}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"✓ Captured photo {capture_count}: {filename}")
                    
                    # Brief pause to show feedback
                    cv2.putText(
                        display_frame,
                        "CAPTURED!",
                        (display_frame.shape[1]//2 - 100, display_frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 255, 0),
                        3,
                    )
                    cv2.imshow("Face Capture", display_frame)
                    cv2.waitKey(500)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(f"Captured {capture_count} photos to {output_dir}")
    print("=" * 70)
    
    if capture_count == 0:
        print("\n⚠️  No photos captured!")
        return 1
    
    if capture_count < 3:
        print(f"\n⚠️  Only {capture_count} photo(s) captured.")
        print("Recommendation: Capture at least 3-5 photos from different angles for better accuracy.")
    
    print("\nNext step: Enroll using captured photos")
    print(f"Run this command:")
    print(f"  python enrollment.py --employee-id {employee_id} --name \"{name}\" --photos {output_dir}")
    print()
    
    # Ask if they want to enroll now
    enroll_now = input("Enroll now? (y/n): ").strip().lower()
    if enroll_now == 'y':
        print("\nEnrolling...")
        import subprocess
        result = subprocess.run([
            sys.executable,
            "enrollment.py",
            "--employee-id", employee_id,
            "--name", name,
            "--photos", str(output_dir)
        ])
        return result.returncode
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
