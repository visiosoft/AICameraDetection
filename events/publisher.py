"""HTTP event publisher with cooldown debouncing and a background worker."""
from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class EventPublisher:
    def __init__(
        self,
        webhook_url: str,
        api_key: str,
        cooldown_seconds: int,
        stop_event: threading.Event,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        request_timeout: float = 5.0,
    ):
        self.webhook_url = webhook_url
        self.api_key = api_key
        self.cooldown_seconds = cooldown_seconds
        self._stop_event = stop_event
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._request_timeout = request_timeout

        self._queue: queue.Queue[dict] = queue.Queue(maxsize=1000)
        self._cooldown: dict[str, float] = {}
        self._cooldown_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="event-publisher", daemon=True
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def publish(
        self,
        employee_id: str,
        name: str,
        confidence: float,
        event_type: str = "recognition",
    ) -> bool:
        now = time.time()
        with self._cooldown_lock:
            last = self._cooldown.get(employee_id, 0.0)
            if now - last < self.cooldown_seconds:
                logger.debug(
                    "Cooldown active for %s (%.1fs remaining)",
                    employee_id,
                    self.cooldown_seconds - (now - last),
                )
                return False
            self._cooldown[employee_id] = now

        payload = {
            "employee_id": employee_id,
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "confidence": round(float(confidence), 4),
            "event_type": event_type,
        }
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            logger.warning("Event queue full, dropping event for %s", employee_id)
            return False
        return True

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._send(payload)

        # Drain remaining events on shutdown.
        while True:
            try:
                payload = self._queue.get_nowait()
            except queue.Empty:
                break
            self._send(payload)

        logger.info("Event publisher thread exiting")

    def _send(self, payload: dict) -> None:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=self._request_timeout,
                )
                if 200 <= resp.status_code < 300:
                    logger.info("Event sent: %s", payload.get("employee_id"))
                    return
                logger.warning(
                    "Backend responded %d for %s (attempt %d/%d)",
                    resp.status_code,
                    payload.get("employee_id"),
                    attempt,
                    self._max_retries,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "POST failed for %s (attempt %d/%d): %s",
                    payload.get("employee_id"),
                    attempt,
                    self._max_retries,
                    exc,
                )
            if attempt < self._max_retries:
                time.sleep(self._retry_backoff * attempt)
        logger.error("Giving up on event for %s", payload.get("employee_id"))
