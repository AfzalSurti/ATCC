"""Thread-safe live stats store for the Streamlit dashboard and stream runner."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class LiveSnapshot:
    """Latest annotated frame and counting stats for UI consumers."""

    frame_bgr: np.ndarray | None = None
    total_events: int = 0
    lane_counts: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    fps: float = 0.0
    timestamp: datetime | None = None
    status: str = "idle"
    report_path: str = ""


class StatsStore:
    """Process-wide store updated by ``Pipeline`` and read by the dashboard."""

    def __init__(self, recent_limit: int = 50) -> None:
        """Create an empty store.

        Args:
            recent_limit: Max recent count events retained for the UI.
        """
        self._lock = threading.Lock()
        self._snap = LiveSnapshot()
        self._recent_limit = recent_limit

    def set_status(self, status: str) -> None:
        """Update pipeline status string (idle/running/error/stopped)."""
        with self._lock:
            self._snap.status = status

    def update_frame(
        self,
        frame_bgr: np.ndarray,
        total_events: int,
        lane_counts: dict[str, int],
        class_counts: dict[str, int],
        fps: float,
        timestamp: datetime,
        report_path: str = "",
    ) -> None:
        """Publish a new annotated frame and aggregate counters."""
        with self._lock:
            self._snap.frame_bgr = frame_bgr.copy()
            self._snap.total_events = total_events
            self._snap.lane_counts = dict(lane_counts)
            self._snap.class_counts = dict(class_counts)
            self._snap.fps = fps
            self._snap.timestamp = timestamp
            self._snap.status = "running"
            if report_path:
                self._snap.report_path = report_path

    def push_event(self, event: dict[str, Any]) -> None:
        """Append a recent count event (oldest dropped past the limit)."""
        with self._lock:
            self._snap.recent_events.append(event)
            if len(self._snap.recent_events) > self._recent_limit:
                self._snap.recent_events = self._snap.recent_events[-self._recent_limit :]

    def snapshot(self) -> LiveSnapshot:
        """Return a shallow copy safe for UI reading."""
        with self._lock:
            frame = None if self._snap.frame_bgr is None else self._snap.frame_bgr.copy()
            return LiveSnapshot(
                frame_bgr=frame,
                total_events=self._snap.total_events,
                lane_counts=dict(self._snap.lane_counts),
                class_counts=dict(self._snap.class_counts),
                recent_events=list(self._snap.recent_events),
                fps=self._snap.fps,
                timestamp=self._snap.timestamp,
                status=self._snap.status,
                report_path=self._snap.report_path,
            )


# Shared singleton used by pipeline + Streamlit in the same process.
GLOBAL_STATS = StatsStore()
