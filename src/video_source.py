"""Video source: file / RTSP frame iterator with configurable sampling.

Phase 1 implements file input fully. RTSP is scaffolded for Phase 3.
"""

from __future__ import annotations

import logging
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


class VideoSource:
    """Open a video file or RTSP URL and yield sampled frames.

    Frame sampling targets ``frame_sampling.target_fps`` so inference runs at a
    stable rate (default ~12 FPS) regardless of source FPS.
    """

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

        if self.source_type == "rtsp":
            # Phase 3 — live stream support
            raise NotImplementedError(
                "RTSP/live stream support is Phase 3. "
                "Use source.type: file for Phase 1 MVP."
            )

        if self.source_type != "file":
            raise ValueError(f"Unsupported source.type: {self.source_type!r}")

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
        stride = max(1, int(round(self.source_fps / self.target_fps)))
        return stride

    def frames(self) -> Generator[FramePacket, None, None]:
        """Yield sampled frames with timestamps derived from FPS + index.

        For files, timestamps are ``epoch_start + frame_index / source_fps``.
        Wall-clock timestamps are reserved for live streams (Phase 3).
        """
        if self._cap is None:
            raise RuntimeError("VideoSource is not open. Call open() first.")

        stride = self._sample_stride()
        logger.info("Frame sampling stride=%d (source_fps=%.2f, target=%.2f)", stride, self.source_fps, self.target_fps)

        # Anchor file timestamps to "now" so Excel intervals look sensible.
        session_start = datetime.now(timezone.utc)
        source_idx = 0
        emitted = 0

        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break

            if source_idx % stride == 0:
                elapsed_sec = source_idx / self.source_fps
                # Use timezone-aware UTC; exporter formats locally-friendly strings.
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

        logger.info("Video exhausted after %d sampled frames (%d source frames)", emitted, source_idx)

    def __iter__(self) -> Iterator[FramePacket]:
        return self.frames()


class LiveVideoSource(VideoSource):
    """Phase 3 stub — RTSP / live camera source with wall-clock timestamps.

    TODO(Phase 3): Implement RTSP reconnect, buffering, and wall-clock stamps.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Stub initializer — raises until Phase 3."""
        raise NotImplementedError(
            "LiveVideoSource is a Phase 3 stub. See docs/TECHNICAL_DESIGN.md."
        )

    def frames(self) -> Generator[FramePacket, None, None]:
        """Yield live frames with wall-clock timestamps (not implemented)."""
        raise NotImplementedError("LiveVideoSource.frames — Phase 3")
        yield  # pragma: no cover — makes this a generator for type checkers
