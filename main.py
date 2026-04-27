"""AI service entry point.

Connects to an RTSP stream, detects faces, tracks them, recognizes them
against a local embedding DB, and POSTs attendance events to a webhook.
"""
from __future__ import annotations

import logging
import signal
import sys
import threading
import time

import config  # noqa: F401  (sets OPENCV_FFMPEG_CAPTURE_OPTIONS before cv2 import)
from config import resolve_gpu, settings, setup_logging

import cv2

from camera.stream import RTSPStream
from database.embeddings import EmbeddingDB
from detection.detector import FaceDetector
from events.publisher import EventPublisher
from recognition.recognizer import FaceRecognizer
from tracking.tracker import IoUTracker, Track

logger = logging.getLogger("ai-service")

WINDOW_NAME = "face-recognition (debug)"


def _draw_overlay(frame, tracks: list[Track]) -> None:
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        if t.recognized_employee_id:
            color = (0, 200, 0)
            label = f"{t.recognized_name} ({t.recognized_conf:.2f})"
        elif t.stable_recognized:
            color = (0, 0, 220)
            label = "Unknown"
        else:
            color = (200, 200, 0)
            label = f"track {t.track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        ty = max(0, y1 - 8)
        cv2.putText(
            frame,
            label,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    setup_logging()
    logger.info("Starting AI service")

    stop_event = threading.Event()

    def _handle_sigint(_signum, _frame):
        logger.info("Shutdown signal received")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        signal.signal(signal.SIGTERM, _handle_sigint)
    except (AttributeError, ValueError):
        pass

    use_gpu = resolve_gpu()
    logger.info("Loading models...")

    detector = FaceDetector(
        conf_threshold=settings.detection_confidence,
        use_gpu=use_gpu,
    )
    recognizer = FaceRecognizer(use_gpu=use_gpu)
    tracker = IoUTracker()

    db = EmbeddingDB(settings.db_path)
    employees = db.list_all()
    logger.info("Loaded %d employees from database", len(employees))

    publisher = EventPublisher(
        webhook_url=settings.backend_webhook_url,
        api_key=settings.backend_api_key,
        cooldown_seconds=settings.cooldown_seconds,
        stop_event=stop_event,
    )
    publisher.start()

    stream = RTSPStream(settings.rtsp_url, stop_event)
    stream.start()

    logger.info("Processing started")
    print(f"\n{'='*60}")
    print(f"Face Recognition Service Running")
    print(f"{'='*60}")
    print(f"Video stream: {settings.rtsp_url}")
    print(f"Enrolled users: {len(employees)}")
    print(f"Recognition threshold: {settings.recognition_threshold}")
    print(f"Detection confidence: {settings.detection_confidence}")
    print(f"{'='*60}\n")

    frame_counter = 0
    fps_window_start = time.monotonic()
    fps_frames = 0
    last_heartbeat = 0

    try:
        while not stop_event.is_set():
            is_new, frame = stream.read_latest()
            if not is_new or frame is None:
                time.sleep(0.005)
                continue

            frame_counter += 1
            fps_frames += 1

            # Heartbeat to show system is running
            if frame_counter - last_heartbeat >= 100:
                print(f"[Heartbeat] Processed {frame_counter} frames, system running...")
                last_heartbeat = frame_counter

            if frame_counter % settings.frame_skip != 0:
                if settings.debug:
                    cv2.imshow(WINDOW_NAME, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_event.set()
                continue

            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            # Show detection/tracking status
            if len(detections) > 0:
                print(f"[Frame {frame_counter}] Detected {len(detections)} face(s), tracking {len(tracks)}")

            recognized_this_frame = 0
            unknown_this_frame = 0
            for t in tracks:
                if not tracker.needs_recognition(t):
                    continue
                print(f"  → Recognizing track {t.track_id} (stable for {t.hits} frames)...")
                emb = recognizer.embed(frame, t.bbox)
                t.stable_recognized = True
                if emb is None:
                    print(f"  ✗ Failed to extract face embedding for track {t.track_id}")
                    continue
                match = recognizer.match(
                    emb, employees, settings.recognition_threshold
                )
                if match is None:
                    unknown_this_frame += 1
                    print(f"  ✗ Unknown person (no match in database)")
                    continue
                t.recognized_employee_id = match.employee_id
                t.recognized_name = match.name
                t.recognized_conf = match.score
                recognized_this_frame += 1
                # Console output only for enrolled users
                print(f"  ✓ {match.name} detected (ID: {match.employee_id}, confidence: {match.score:.2f})")
                publisher.publish(
                    employee_id=match.employee_id,
                    name=match.name,
                    confidence=match.score,
                )



            if settings.debug:
                _draw_overlay(frame, tracks)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()

                if fps_frames >= 30:
                    elapsed = time.monotonic() - fps_window_start
                    fps = fps_frames / elapsed if elapsed > 0 else 0.0
                    logger.debug("Processing FPS: %.1f", fps)
                    fps_window_start = time.monotonic()
                    fps_frames = 0
    finally:
        logger.info("Shutting down")
        stop_event.set()
        stream.join(timeout=5.0)
        publisher.join(timeout=5.0)
        db.close()
        if settings.debug:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        logging.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
