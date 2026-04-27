"""Verify enrollment and test recognition against enrolled users."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from config import resolve_gpu, settings, setup_logging

import cv2
import numpy as np

from database.embeddings import EmbeddingDB
from recognition.recognizer import FaceRecognizer


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def main() -> int:
    setup_logging()
    
    print("\n" + "=" * 70)
    print("ENROLLMENT VERIFICATION TEST")
    print("=" * 70)
    print(f"Database path: {settings.db_path}")
    print(f"Recognition threshold: {settings.recognition_threshold}")
    print(f"Using GPU: {resolve_gpu()}")
    print("=" * 70 + "\n")

    # Check database
    db = EmbeddingDB(settings.db_path)
    employees = db.list_all()
    
    if not employees:
        print("❌ No employees found in database!")
        print(f"\nTo enroll a user, run:")
        print(f"  python enrollment.py --employee-id TEST001 --name 'Test User' --photos photos/test_user")
        db.close()
        return 1
    
    print(f"✓ Found {len(employees)} enrolled employee(s):\n")
    for emp in employees:
        print(f"  • ID: {emp.employee_id}")
        print(f"    Name: {emp.name}")
        print(f"    Enrolled: {emp.enrolled_at}")
        print(f"    Embedding shape: {emp.embedding.shape}")
        print()
    
    # Test recognition
    photo_path = input("\nEnter path to test photo (or press Enter to skip): ").strip()
    
    if not photo_path:
        print("\nSkipping recognition test.")
        db.close()
        return 0
    
    if not Path(photo_path).exists():
        print(f"❌ File not found: {photo_path}")
        db.close()
        return 1
    
    print(f"\nLoading image: {photo_path}")
    img = cv2.imread(photo_path)
    if img is None:
        print("❌ Failed to read image")
        db.close()
        return 1
    
    print("Loading recognition model...")
    use_gpu = resolve_gpu()
    recognizer = FaceRecognizer(use_gpu=use_gpu)
    
    print("Extracting embedding from test photo...")
    test_embedding = recognizer.embed_from_full_frame(img)
    
    if test_embedding is None:
        print("❌ No face detected in test photo")
        db.close()
        return 1
    
    print("✓ Face detected and embedding extracted")
    print(f"  Test embedding shape: {test_embedding.shape}")
    
    # Test matching
    print("\n" + "-" * 70)
    print("MATCHING RESULTS:")
    print("-" * 70)
    
    test_emb_norm = _l2_normalize(np.asarray(test_embedding, dtype=np.float32))
    
    best_match = None
    best_score = -1.0
    
    for emp in employees:
        emp_emb_norm = _l2_normalize(np.asarray(emp.embedding, dtype=np.float32))
        similarity = float(np.dot(test_emb_norm, emp_emb_norm))
        
        status = "✓ MATCH" if similarity >= settings.recognition_threshold else "✗ No match"
        print(f"\n{emp.employee_id} ({emp.name})")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Threshold: {settings.recognition_threshold:.4f}")
        print(f"  Status: {status}")
        
        if similarity > best_score:
            best_score = similarity
            best_match = emp
    
    print("\n" + "=" * 70)
    if best_match and best_score >= settings.recognition_threshold:
        print(f"✓ RECOGNIZED: {best_match.name} (ID: {best_match.employee_id})")
        print(f"  Confidence: {best_score:.4f}")
    else:
        print("✗ NO MATCH FOUND")
        if best_match:
            print(f"  Best similarity was {best_score:.4f} with {best_match.name}")
            print(f"  Below threshold of {settings.recognition_threshold:.4f}")
            print("\nTip: You may need to:")
            print("  1. Lower recognition threshold in .env")
            print("  2. Re-enroll with better quality photos")
            print("  3. Use multiple photos from different angles")
    print("=" * 70 + "\n")
    
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
