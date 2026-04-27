"""Smoke test for the RTSP stream reader. Run from the ai-service folder."""
from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # sets RTSP env  # noqa: F401
from config import settings, setup_logging

import cv2

from camera.stream import RTSPStream

WINDOW = "test_camera"


def main() -> int:
    setup_logging("INFO")
    print(f"Connecting to {settings.rtsp_url}")
    stop_event = threading.Event()
    stream = RTSPStream(settings.rtsp_url, stop_event)
    stream.start()

    deadline = time.monotonic() + 10.0
    frames_seen = 0
    last_id = -1

    try:
        while time.monotonic() < deadline:
            is_new, frame = stream.read_latest()
            if is_new and frame is not None:
                frames_seen += 1
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        stream.join(timeout=2.0)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print(f"Frames received in 10s: {frames_seen}")
    if frames_seen == 0:
        print("No frame received. Is MediaMTX streaming and reachable?")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
