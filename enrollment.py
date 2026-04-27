"""CLI tool to enroll, list, and delete employee face embeddings."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import config  # sets RTSP env / loads settings
from config import resolve_gpu, settings, setup_logging

import numpy as np

from database.embeddings import EmbeddingDB

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _list_photos(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in VALID_EXTS:
            out.append(os.path.join(folder, name))
    return out


def cmd_enroll(args: argparse.Namespace) -> int:
    import cv2  # imported after config so RTSP env is set
    from recognition.recognizer import FaceRecognizer

    photos = _list_photos(args.photos)
    if not photos:
        print(f"No images found in {args.photos}", file=sys.stderr)
        return 1

    use_gpu = resolve_gpu()
    recognizer = FaceRecognizer(use_gpu=use_gpu)

    embeddings: List[np.ndarray] = []
    for path in photos:
        name = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"Processing {name}... failed to read image")
            continue
        emb = recognizer.embed_from_full_frame(img)
        if emb is None:
            print(f"Processing {name}... no face detected")
            continue
        embeddings.append(emb)
        print(f"Processing {name}... face detected, embedding generated")

    if not embeddings:
        print("No usable embeddings produced. Aborting.", file=sys.stderr)
        return 1

    avg = _l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32))

    db = EmbeddingDB(settings.db_path)
    try:
        db.upsert(args.employee_id, args.name, avg)
    finally:
        db.close()

    print(f"Enrolled employee {args.employee_id} with {len(embeddings)} embeddings")
    print(f"Saved to {settings.db_path}")
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    db = EmbeddingDB(settings.db_path)
    try:
        records = db.list_all()
    finally:
        db.close()

    if not records:
        print("No employees enrolled.")
        return 0

    print(f"{'EMPLOYEE_ID':<20} {'NAME':<30} {'ENROLLED_AT'}")
    print("-" * 75)
    for r in records:
        print(f"{r.employee_id:<20} {r.name:<30} {r.enrolled_at}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    db = EmbeddingDB(settings.db_path)
    try:
        deleted = db.delete(args.delete)
    finally:
        db.close()
    if deleted:
        print(f"Deleted employee {args.delete}")
        return 0
    print(f"No employee found with ID {args.delete}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="enrollment",
        description="Enroll, list, or delete employee face embeddings.",
    )
    p.add_argument("--employee-id", help="Unique ID for the employee")
    p.add_argument("--name", help="Display name for the employee")
    p.add_argument("--photos", help="Folder containing photos of the employee")
    p.add_argument("--list", action="store_true", help="List all enrolled employees")
    p.add_argument("--delete", metavar="EMPLOYEE_ID", help="Delete an employee by ID")
    return p


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    actions = sum(
        1
        for x in (args.list, args.delete, args.employee_id or args.name or args.photos)
        if x
    )
    if actions == 0:
        parser.print_help()
        return 2

    if args.list:
        return cmd_list(args)
    if args.delete:
        return cmd_delete(args)

    missing = [
        flag
        for flag, val in (
            ("--employee-id", args.employee_id),
            ("--name", args.name),
            ("--photos", args.photos),
        )
        if not val
    ]
    if missing:
        parser.error(f"missing required argument(s): {', '.join(missing)}")
    return cmd_enroll(args)


if __name__ == "__main__":
    sys.exit(main())
