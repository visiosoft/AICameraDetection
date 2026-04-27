"""Threaded RTSP frame reader.

Always exposes the most recent decoded frame; older frames are dropped so the
recognition pipeline never falls behind a queued backlog. Reconnects with
exponential backoff if the stream drops.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RTSPStream:
    def __init__(
        self,
        url: str,
        stop_event: threading.Event,
        reconnect_max_backoff: float = 30.0,
    ):
        self.url = url
        self._stop_event = stop_event
        self._reconnect_max_backoff = reconnect_max_backoff
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._consumed_id: int = 0
        self._connected = False
        self._thread: Optional[threading.Thread] = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="rtsp-reader", daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (is_new, frame). is_new is False if the caller already saw this frame."""
        with self._lock:
            if self._latest_frame is None:
                return False, None
            is_new = self._frame_id != self._consumed_id
            self._consumed_id = self._frame_id
            return is_new, self._latest_frame.copy()

    def _open(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop_event.is_set():
            logger.info("Connecting to %s", self.url)
            cap = self._open()
            if cap is None:
                self._connected = False
                logger.warning(
                    "Failed to open RTSP stream, retrying in %.1fs", backoff
                )
                self._sleep_with_stop(backoff)
                backoff = min(backoff * 2, self._reconnect_max_backoff)
                continue

            self._connected = True
            backoff = 1.0
            logger.info("Camera stream connected")

            consecutive_failures = 0
            while not self._stop_event.is_set():
                grabbed = cap.grab()
                if not grabbed:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        logger.warning("Lost RTSP stream, reconnecting")
                        break
                    time.sleep(0.05)
                    continue
                consecutive_failures = 0

                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    continue

                with self._lock:
                    self._latest_frame = frame
                    self._frame_id += 1

            cap.release()
            self._connected = False

        logger.info("RTSP reader thread exiting")

    def _sleep_with_stop(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.1)
