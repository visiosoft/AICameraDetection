"""Threaded RTSP frame reader using FFmpeg subprocess.

Pipes raw BGR24 frames from FFmpeg directly, avoiding OpenCV's RTSP overhead.
Always exposes the most recent decoded frame; older frames are dropped so the
recognition pipeline never falls behind a queued backlog. Reconnects with
exponential backoff if the stream drops.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default resolution — overridden by probing the stream on first connect.
_DEFAULT_WIDTH = 1920
_DEFAULT_HEIGHT = 1080


def _find_ffmpeg() -> str:
    """Locate the ffmpeg binary, preferring PATH."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    raise FileNotFoundError(
        "ffmpeg not found on PATH. Install FFmpeg and ensure it is accessible."
    )


def _probe_resolution(ffmpeg: str, url: str, timeout: float = 10.0) -> Tuple[int, int]:
    """Use ffprobe to discover the stream's width and height."""
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    if not shutil.which(ffprobe):
        ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-v", "error",
        "-rtsp_transport", "tcp",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        url,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        parts = result.stdout.strip().split("x")
        if len(parts) == 2:
            w, h = int(parts[0]), int(parts[1])
            if w > 0 and h > 0:
                return w, h
    except Exception as exc:
        logger.debug("ffprobe resolution detection failed: %s", exc)
    return _DEFAULT_WIDTH, _DEFAULT_HEIGHT


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
        self._ffmpeg = _find_ffmpeg()
        self._width = 0
        self._height = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._width, self._height

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

    def _build_ffmpeg_cmd(self) -> list[str]:
        return [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            # Low-latency input options
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "500000",   # 0.5s — fast stream analysis
            "-probesize", "500000",         # 500KB probe
            "-rtsp_transport", "tcp",
            "-i", self.url,
            # Output: raw BGR24 frames to stdout
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",                          # no audio
            "-sn",                          # no subtitles
            "pipe:1",
        ]

    def _run(self) -> None:
        backoff = 1.0

        # Probe resolution once
        logger.info("Probing stream resolution for %s", self.url)
        self._width, self._height = _probe_resolution(self._ffmpeg, self.url)
        logger.info("Stream resolution: %dx%d", self._width, self._height)
        frame_bytes = self._width * self._height * 3

        while not self._stop_event.is_set():
            logger.info("Connecting to %s via FFmpeg", self.url)
            cmd = self._build_ffmpeg_cmd()
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=frame_bytes,
                )
            except OSError as exc:
                self._connected = False
                logger.warning("Failed to start FFmpeg: %s, retrying in %.1fs", exc, backoff)
                self._sleep_with_stop(backoff)
                backoff = min(backoff * 2, self._reconnect_max_backoff)
                continue

            self._connected = True
            backoff = 1.0
            logger.info("Camera stream connected (FFmpeg PID %d)", proc.pid)

            try:
                while not self._stop_event.is_set():
                    raw = proc.stdout.read(frame_bytes)
                    if len(raw) != frame_bytes:
                        # Stream ended or partial read → reconnect
                        logger.warning("RTSP stream interrupted, reconnecting")
                        break
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self._height, self._width, 3)
                    )
                    with self._lock:
                        self._latest_frame = frame
                        self._frame_id += 1
            finally:
                # Clean shutdown of FFmpeg process
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                self._connected = False

            if not self._stop_event.is_set():
                # Read stderr for diagnostics
                try:
                    stderr = proc.stderr.read().decode(errors="replace").strip()
                    if stderr:
                        logger.debug("FFmpeg stderr: %s", stderr[:500])
                except Exception:
                    pass
                self._sleep_with_stop(backoff)
                backoff = min(backoff * 2, self._reconnect_max_backoff)

        logger.info("RTSP reader thread exiting")

    def _sleep_with_stop(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.1)
