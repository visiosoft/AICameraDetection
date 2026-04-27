"""Application configuration.

This module sets the OPENCV_FFMPEG_CAPTURE_OPTIONS environment variable at
import time so it takes effect before any other module imports cv2. Always
import config (or something that imports it) before importing cv2.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Literal

# Must be set BEFORE any cv2 import anywhere in the process.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    rtsp_url: str = Field(default="rtsp://localhost:8554/test")
    backend_webhook_url: str = Field(
        default="http://localhost:3000/api/attendance/events"
    )
    backend_api_key: str = Field(default="")
    detection_confidence: float = Field(default=0.5)
    recognition_threshold: float = Field(default=0.5)
    frame_skip: int = Field(default=3)
    db_path: str = Field(default="./data/employees.db")
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)
    cooldown_seconds: int = Field(default=60)
    use_gpu: Literal["auto", "true", "false"] = Field(default="auto")
    recognition_buffer_size: int = Field(default=5)

    @field_validator("detection_confidence", "recognition_threshold")
    @classmethod
    def _clamp_unit(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("frame_skip")
    @classmethod
    def _frame_skip_min(cls, v: int) -> int:
        return max(1, int(v))

    @field_validator("cooldown_seconds")
    @classmethod
    def _cooldown_min(cls, v: int) -> int:
        return max(0, int(v))

    @field_validator("recognition_buffer_size")
    @classmethod
    def _buffer_size_min(cls, v: int) -> int:
        return max(1, int(v))

    @field_validator("log_level")
    @classmethod
    def _log_level_upper(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            return "INFO"
        return v

    @field_validator("use_gpu", mode="before")
    @classmethod
    def _normalize_use_gpu(cls, v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        s = str(v).strip().lower()
        if s in {"auto", "true", "false"}:
            return s
        if s in {"1", "yes", "on"}:
            return "true"
        if s in {"0", "no", "off"}:
            return "false"
        return "auto"

    @field_validator("db_path")
    @classmethod
    def _resolve_db_path(cls, v: str) -> str:
        """Convert relative db_path to absolute path relative to this config file."""
        if os.path.isabs(v):
            return v
        # Resolve relative to the directory containing config.py (ai-service/)
        config_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(config_dir, v))


settings = Settings()


def resolve_gpu() -> bool:
    """Decide whether to run on GPU based on settings and runtime availability."""
    if settings.use_gpu == "false":
        return False
    if settings.use_gpu == "true":
        return True

    # auto
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return True
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    return False


_LOGGING_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    lvl = getattr(logging, (level or settings.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    # Quiet noisy third-party loggers.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _LOGGING_CONFIGURED = True
