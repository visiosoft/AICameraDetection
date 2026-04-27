"""Diagnose video stream and face detection issues."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from config import resolve_gpu, settings, setup_logging

import cv2
import numpy as np

from detection.detector import FaceDetector


def main() -> int:
    setup_logging()
    
    print("\n" + "=" * 70)
    print("VIDEO STREAM DIAGNOSTIC")
    print("=" * 70)
    print(f"RTSP URL: {settings.rtsp_url}")
    print(f"Detection confidence: {settings.detection_confidence}")
    print(f"Frame skip: {settings.frame_skip}")
    print(f"Using GPU: {resolve_gpu()}")
    print("=" * 70 + "\n")

    # Test 1: Connect to stream
    print("Test 1: Connecting to video stream...")
    cap = cv2.VideoCapture(settings.rtsp_url)
    
    if not cap.isOpened():
        print(f"❌ FAILED: Cannot connect to {settings.rtsp_url}")
        print("\nTroubleshooting:")
        print("  1. Check if MediaMTX is running")
        print("  2. Verify the RTSP URL is correct")
        print("  3. Test with VLC: vlc rtsp://localhost:8554/test")
        return 1
    
    print("✓ PASSED: Connected to stream\n")
    
    # Test 2: Read frames
    print("Test 2: Reading frames from stream...")
    frame_count = 0
    read_failures = 0
    
    for i in range(30):  # Try 30 frames
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_count += 1
            if frame_count == 1:
                h, w = frame.shape[:2]
                print(f"✓ PASSED: Successfully reading frames")
                print(f"  Frame size: {w}x{h}")
                print(f"  Frame shape: {frame.shape}")
        else:
            read_failures += 1
        time.sleep(0.03)
    
    if frame_count == 0:
        print("❌ FAILED: Could not read any frames")
        cap.release()
        return 1
    
    print(f"  Successfully read {frame_count}/30 frames\n")
    
    # Test 3: Face detection
    print("Test 3: Testing face detection...")
    print("Loading detector model...")
    use_gpu = resolve_gpu()
    detector = FaceDetector(
        conf_threshold=settings.detection_confidence,
        use_gpu=use_gpu,
    )
    print("✓ Detector loaded\n")
    
    print("Analyzing 100 frames for faces...")
    detection_results = []
    frames_with_faces = 0
    total_faces = 0
    
    for i in range(100):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        detections = detector.detect(frame)
        detection_results.append(len(detections))
        
        if len(detections) > 0:
            frames_with_faces += 1
            total_faces += len(detections)
            if frames_with_faces <= 5:  # Show first 5 detections
                print(f"  Frame {i}: Detected {len(detections)} face(s)")
                for j, det in enumerate(detections):
                    x1, y1, x2, y2 = det.bbox
                    w = x2 - x1
                    h = y2 - y1
                    print(f"    Face {j+1}: size={w}x{h}px, confidence={det.conf:.3f}")
    
    cap.release()
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC RESULTS")
    print("=" * 70)
    print(f"Total frames analyzed: 100")
    print(f"Frames with faces: {frames_with_faces}")
    print(f"Total faces detected: {total_faces}")
    print(f"Detection rate: {frames_with_faces}%")
    
    if frames_with_faces == 0:
        print("\n⚠️  NO FACES DETECTED!")
        print("\nPossible issues:")
        print("  1. Face is too small in the frame (try moving closer to camera)")
        print("  2. Video quality is too low (blurry, dark, or pixelated)")
        print("  3. Face angle is not frontal (try facing camera directly)")
        print("  4. Detection confidence is too high (currently: {:.2f})".format(settings.detection_confidence))
        print("\nSuggestions:")
        print("  • Test with a high-quality image first:")
        print("    python tests/test_detector.py")
        print("  • Lower detection confidence in .env:")
        print("    DETECTION_CONFIDENCE=0.2")
        print("  • Check video with:")
        print("    vlc " + settings.rtsp_url)
    else:
        print(f"\n✓ Face detection is working!")
        if frames_with_faces < 50:
            print(f"\n⚠️  Low detection rate ({frames_with_faces}%)")
            print("Consider:")
            print("  • Improving lighting conditions")
            print("  • Moving closer to camera")
            print("  • Ensuring face is clearly visible")
    
    print("=" * 70 + "\n")
    
    return 0 if frames_with_faces > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
