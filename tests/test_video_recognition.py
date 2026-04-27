"""Test if video frames can be recognized against enrolled embeddings."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from config import resolve_gpu, settings, setup_logging

import cv2
import numpy as np

from database.embeddings import EmbeddingDB
from detection.detector import FaceDetector
from recognition.recognizer import FaceRecognizer


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def main() -> int:
    setup_logging()
    
    print("\n" + "=" * 70)
    print("VIDEO RECOGNITION TEST")
    print("=" * 70)
    print(f"RTSP URL: {settings.rtsp_url}")
    print(f"Recognition threshold: {settings.recognition_threshold}")
    print("=" * 70 + "\n")

    # Load enrolled users
    db = EmbeddingDB(settings.db_path)
    employees = db.list_all()
    
    if not employees:
        print("❌ No enrolled users found!")
        db.close()
        return 1
    
    print(f"✓ Loaded {len(employees)} enrolled user(s):")
    for emp in employees:
        print(f"  • {emp.employee_id}: {emp.name}")
    print()

    # Connect to stream
    print("Connecting to video stream...")
    cap = cv2.VideoCapture(settings.rtsp_url)
    
    if not cap.isOpened():
        print(f"❌ Cannot connect to {settings.rtsp_url}")
        db.close()
        return 1
    
    print("✓ Connected\n")

    # Load models
    print("Loading models...")
    use_gpu = resolve_gpu()
    detector = FaceDetector(conf_threshold=settings.detection_confidence, use_gpu=use_gpu)
    recognizer = FaceRecognizer(use_gpu=use_gpu)
    print("✓ Models loaded\n")

    print("=" * 70)
    print("Analyzing video frames for recognition...")
    print("Checking first 50 frames with detected faces")
    print("=" * 70 + "\n")

    analyzed_faces = 0
    frames_checked = 0
    recognition_attempts = 0
    successful_matches = 0
    
    while analyzed_faces < 50 and frames_checked < 200:
        ret, frame = cap.read()
        frames_checked += 1
        
        if not ret or frame is None:
            continue
        
        detections = detector.detect(frame)
        
        if not detections:
            continue
        
        # Test recognition on each detected face
        for det in detections:
            analyzed_faces += 1
            recognition_attempts += 1
            
            print(f"\n[Face {analyzed_faces}] Frame {frames_checked}")
            print(f"  BBox: {det.bbox}, Detection confidence: {det.conf:.3f}")
            
            # Extract embedding
            emb = recognizer.embed(frame, det.bbox, margin=0.2)
            
            if emb is None:
                print(f"  ✗ Failed to extract embedding from video frame")
                continue
            
            print(f"  ✓ Embedding extracted (shape: {emb.shape})")
            
            # Try matching against all enrolled users
            emb_norm = _l2_normalize(np.asarray(emb, dtype=np.float32))
            
            best_match = None
            best_score = -1.0
            
            for emp in employees:
                emp_emb_norm = _l2_normalize(np.asarray(emp.embedding, dtype=np.float32))
                similarity = float(np.dot(emb_norm, emp_emb_norm))
                
                print(f"    vs {emp.employee_id} ({emp.name}): similarity = {similarity:.4f}", end="")
                
                if similarity >= settings.recognition_threshold:
                    print(f" ✓ MATCH")
                    if similarity > best_score:
                        best_score = similarity
                        best_match = emp
                else:
                    print(f" (below threshold {settings.recognition_threshold:.4f})")
            
            if best_match:
                successful_matches += 1
                print(f"  🎯 RECOGNIZED: {best_match.name} (confidence: {best_score:.4f})")
            else:
                print(f"  ✗ No match found")
            
            if analyzed_faces >= 50:
                break
    
    cap.release()
    db.close()

    print("\n" + "=" * 70)
    print("RECOGNITION TEST RESULTS")
    print("=" * 70)
    print(f"Frames checked: {frames_checked}")
    print(f"Faces analyzed: {analyzed_faces}")
    print(f"Recognition attempts: {recognition_attempts}")
    print(f"Successful matches: {successful_matches}")
    print(f"Match rate: {100*successful_matches/recognition_attempts if recognition_attempts > 0 else 0:.1f}%")
    
    if successful_matches == 0:
        print("\n⚠️  NO MATCHES FOUND!")
        print("\nPossible reasons:")
        print("  1. Video quality is too different from enrollment photo")
        print("  2. Lighting/angle in video differs from enrollment")
        print("  3. Recognition threshold is too strict (current: {:.2f})".format(settings.recognition_threshold))
        print("  4. Need to re-enroll using video frames instead of photos")
        print("\nSuggestions:")
        print("  • Lower recognition threshold in .env:")
        print("    RECOGNITION_THRESHOLD=0.25")
        print("  • Re-enroll using a frame capture from the video:")
        print("    1. Capture frame from video")
        print("    2. Save as photo")
        print("    3. Re-run enrollment with that photo")
    elif successful_matches < recognition_attempts * 0.5:
        print(f"\n⚠️  Low match rate ({100*successful_matches/recognition_attempts:.1f}%)")
        print("Consider lowering recognition threshold or re-enrolling with video frames")
    else:
        print(f"\n✓ Recognition is working! Match rate: {100*successful_matches/recognition_attempts:.1f}%")
    
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
