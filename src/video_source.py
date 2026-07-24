"""Video source: file / RTSP frame iterator with configurable sampling."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generator, Iterator

import cv2
import numpy as np

from src.config_loader import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class FramePacket:
    """A single sampled frame with timing metadata."""

    frame: np.ndarray
    frame_index: int
    source_frame_index: int
    timestamp: datetime
    source_fps: float


def create_video_source(config: dict[str, Any]) -> "VideoSource":
    """Factory returning a file or live RTSP video source.

    Args:
        config: Full pipeline config.

    Returns:
        ``VideoSource`` or ``LiveVideoSource``.
    """
    source_type = str(config.get("source", {}).get("type", "file")).lower()
    if source_type in {"rtsp", "live", "stream", "webcam"}:
        return LiveVideoSource(config)
    if source_type == "file":
        return VideoSource(config)
    raise ValueError(f"Unsupported source.type: {source_type!r}")


class VideoSource:
    """Open a video file and yield sampled frames."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize from the top-level pipeline config.

        Args:
            config: Full ``camera_config.yaml`` dictionary.
        """
        source_cfg = config.get("source", {})
        sampling_cfg = config.get("frame_sampling", {})

        self.source_type: str = str(source_cfg.get("type", "file")).lower()
        self.raw_path: str = str(source_cfg.get("path", ""))
        self.target_fps: float = float(sampling_cfg.get("target_fps", 12))
        self._cap: cv2.VideoCapture | None = None
        self.source_fps: float = 0.0
        self.frame_count: int = 0
        self.width: int = 0
        self.height: int = 0

    def open(self) -> None:
        """Open the video capture and read stream metadata."""
        path = resolve_path(self.raw_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Video file not found: {path}. "
                "Place a sample under data/sample_videos/ and update "
                "config/camera_config.yaml source.path."
            )

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

        self.source_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.source_fps <= 0:
            logger.warning("Source FPS unavailable; assuming 30.0")
            self.source_fps = 30.0

        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        logger.info(
            "Opened video %s | %dx%d @ %.2f FPS | %d frames | target_fps=%.2f",
            path,
            self.width,
            self.height,
            self.source_fps,
            self.frame_count,
            self.target_fps,
        )

    def close(self) -> None:
        """Release the capture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _sample_stride(self) -> int:
        """Compute how many source frames to skip between processed frames."""
        return max(1, int(round(self.source_fps / self.target_fps)))

    def frames(self) -> Generator[FramePacket, None, None]:
        """Yield sampled frames with timestamps derived from FPS + index."""
        if self._cap is None:
            raise RuntimeError("VideoSource is not open. Call open() first.")

        stride = self._sample_stride()
        logger.info(
            "Frame sampling stride=%d (source_fps=%.2f, target=%.2f)",
            stride,
            self.source_fps,
            self.target_fps,
        )

        session_start = datetime.now(timezone.utc)
        source_idx = 0
        emitted = 0

        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break

            if source_idx % stride == 0:
                elapsed_sec = source_idx / self.source_fps
                ts = datetime.fromtimestamp(
                    session_start.timestamp() + elapsed_sec,
                    tz=timezone.utc,
                )
                yield FramePacket(
                    frame=frame,
                    frame_index=emitted,
                    source_frame_index=source_idx,
                    timestamp=ts,
                    source_fps=self.source_fps,
                )
                emitted += 1

            source_idx += 1

        logger.info(
            "Video exhausted after %d sampled frames (%d source frames)",
            emitted,
            source_idx,
        )

    def __iter__(self) -> Iterator[FramePacket]:
        return self.frames()


class LiveVideoSource(VideoSource):
    """RTSP / webcam live source with wall-clock timestamps and reconnect.

    Config keys under ``source``:
      - type: rtsp | live | stream | webcam
      - path / url: RTSP URL, or webcam index as string (\"0\")
      - reconnect_seconds: wait between reconnect attempts (default 3)
      - read_timeout_ms: OpenCV open/read timeout hint (default 5000)
      - max_reconnects: 0 = unlimited (default 0)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize live capture settings from config."""
        super().__init__(config)
        source_cfg = config.get("source", {})
        self.raw_path = str(
            source_cfg.get("url")
            or source_cfg.get("path")
            or source_cfg.get("rtsp")
            or ""
        )
        if not self.raw_path:
            raise ValueError("Live source requires source.path or source.url (RTSP / webcam index).")

        self.reconnect_seconds: float = float(source_cfg.get("reconnect_seconds", 3))
        self.read_timeout_ms: int = int(source_cfg.get("read_timeout_ms", 5000))
        self.max_reconnects: int = int(source_cfg.get("max_reconnects", 0))
        self._stop = False

    def open(self) -> None:
        """Open RTSP/webcam with low-latency buffer settings."""
        self._open_capture(initial=True)

    def _open_capture(self, initial: bool = False) -> None:
        """(Re)open the capture device/URL."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        target = self.raw_path
        # Webcam index support: path "0" → device 0
        open_arg: str | int = int(target) if target.isdigit() else target

        cap = cv2.VideoCapture(open_arg)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.read_timeout_ms > 0 and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(self.read_timeout_ms))
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(self.read_timeout_ms))
        except Exception:  # noqa: BLE001 — backend-dependent properties
            pass

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open live source: {self.raw_path}")

        self._cap = cap
        self.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.source_fps <= 0:
            self.source_fps = max(self.target_fps, 25.0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_count = 0
        logger.info(
            "%s live source %s | %dx%d @ ~%.2f FPS | target_fps=%.2f",
            "Opened" if initial else "Reconnected",
            self.raw_path,
            self.width,
            self.height,
            self.source_fps,
            self.target_fps,
        )

    def stop(self) -> None:
        """Request the live frame loop to stop."""
        self._stop = True

    def frames(self) -> Generator[FramePacket, None, None]:
        """Yield live frames throttled to ``target_fps`` with wall-clock timestamps."""
        if self._cap is None:
            raise RuntimeError("LiveVideoSource is not open. Call open() first.")

        min_interval = 1.0 / max(self.target_fps, 0.1)
        emitted = 0
        source_idx = 0
        reconnects = 0
        last_emit = 0.0

        while not self._stop:
            assert self._cap is not None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                reconnects += 1
                if self.max_reconnects and reconnects > self.max_reconnects:
                    logger.error("Max reconnects (%d) exceeded — stopping live source", self.max_reconnects)
                    break
                logger.warning(
                    "Live read failed — reconnecting in %.1fs (attempt %d)",
                    self.reconnect_seconds,
                    reconnects,
                )
                time.sleep(self.reconnect_seconds)
                try:
                    self._open_capture(initial=False)
                except RuntimeError as exc:
                    logger.error("Reconnect failed: %s", exc)
                continue

            reconnects = 0
            source_idx += 1
            now = time.monotonic()
            if last_emit and (now - last_emit) < min_interval:
                continue
            last_emit = now

            # Grab freshest frame if buffer still has backlog.
            for _ in range(2):
                peek_ok, peek = self._cap.read()
                if peek_ok and peek is not None:
                    frame = peek
                    source_idx += 1
                else:
                    break

            if self.width == 0 or self.height == 0:
                self.height, self.width = frame.shape[:2]

            yield FramePacket(
                frame=frame,
                frame_index=emitted,
                source_frame_index=source_idx,
                timestamp=datetime.now(timezone.utc),
                source_fps=self.source_fps,
            )
            emitted += 1

        logger.info("Live source stopped after %d sampled frames", emitted)
