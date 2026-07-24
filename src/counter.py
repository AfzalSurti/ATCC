"""Per-lane line-crossing counters with track_id deduplication.

Uses supervision geometry types and LineZone objects (for annotation /
in-out display). Crossing detection uses per-track centroid history with
segment intersection so counting stays correct across NumPy versions
(supervision LineZone.trigger relies on np.cross 2D behaviour removed in
NumPy 2+).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import supervision as sv

from src.schemas import CountEvent, Detection

logger = logging.getLogger(__name__)


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    """Return True if segment p1→p2 intersects segment q1→q2."""

    def orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
        return (
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, p2, q2):
        return True
    if o3 == 0 and on_segment(q1, q2, p1):
        return True
    if o4 == 0 and on_segment(q1, q2, p2):
        return True
    return False


def _centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the center of an xyxy bbox."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class LaneState:
    """Runtime state for one lane's counting line and dedupe set."""

    name: str
    direction: str
    line_start: tuple[float, float]
    line_end: tuple[float, float]
    line_zone: sv.LineZone  # kept for visualization annotators
    counted_track_ids: set[int] = field(default_factory=set)
    counts: dict[str, int] = field(default_factory=dict)
    display_count: int = 0


class LaneCounter:
    """Count vehicles per lane / direction / class using line crossings."""

    def __init__(
        self,
        config: dict[str, Any],
        on_count: Callable[[CountEvent], None] | None = None,
    ) -> None:
        """Build counting lines from ``lanes`` in config.

        Args:
            config: Full pipeline config.
            on_count: Optional callback invoked for each new count event.
        """
        self.on_count = on_count
        self.lanes: list[LaneState] = []
        self._last_centroid: dict[int, tuple[float, float]] = {}
        self._class_names: list[str] = list(
            config.get("classes", {}).get(
                "target", ["bicycle", "motorcycle", "car", "bus", "truck"]
            )
        )

        for lane_cfg in config.get("lanes", []):
            name = str(lane_cfg["name"])
            direction = str(lane_cfg.get("direction", "unknown"))
            line = lane_cfg["line"]
            if len(line) != 2:
                raise ValueError(f"Lane {name}: line must be [[x1,y1],[x2,y2]]")

            start_xy = (float(line[0][0]), float(line[0][1]))
            end_xy = (float(line[1][0]), float(line[1][1]))
            start = sv.Point(start_xy[0], start_xy[1])
            end = sv.Point(end_xy[0], end_xy[1])
            line_zone = sv.LineZone(start=start, end=end)
            self.lanes.append(
                LaneState(
                    name=name,
                    direction=direction,
                    line_start=start_xy,
                    line_end=end_xy,
                    line_zone=line_zone,
                )
            )
            logger.info("Lane %s (%s) line %s → %s", name, direction, line[0], line[1])

    def reset_window_counts(self) -> None:
        """Clear per-lane count mirrors after an aggregator interval rollover.

        Deduped track IDs are retained so a vehicle already counted is not
        double-counted in a later interval if it loiters on the line.
        """
        for lane in self.lanes:
            lane.counts.clear()

    def update(self, detections: list[Detection]) -> list[CountEvent]:
        """Update all lanes; return newly counted crossing events.

        A vehicle is counted once when its track centroid path intersects the
        lane line, deduped by ``track_id``.

        Args:
            detections: Tracked detections (must include ``track_id``).

        Returns:
            List of new ``CountEvent`` objects for this frame.
        """
        events: list[CountEvent] = []
        seen_this_frame: set[int] = set()

        for det in detections:
            if det.track_id is None:
                continue
            tid = det.track_id
            seen_this_frame.add(tid)
            curr = _centroid(det.bbox)
            prev = self._last_centroid.get(tid)
            self._last_centroid[tid] = curr

            if prev is None:
                continue

            for lane in self.lanes:
                if tid in lane.counted_track_ids:
                    continue
                if not _segments_intersect(prev, curr, lane.line_start, lane.line_end):
                    continue

                lane.counted_track_ids.add(tid)
                lane.counts[det.class_name] = lane.counts.get(det.class_name, 0) + 1
                lane.display_count += 1

                event = CountEvent(
                    track_id=tid,
                    lane=lane.name,
                    direction=lane.direction,
                    class_name=det.class_name,
                    class_id=det.class_id,
                )
                events.append(event)
                logger.debug(
                    "COUNT track=%s lane=%s dir=%s class=%s",
                    tid,
                    lane.name,
                    lane.direction,
                    det.class_name,
                )
                if self.on_count is not None:
                    self.on_count(event)

        # Drop centroids for tracks no longer visible (avoid stale jumps).
        stale = [tid for tid in self._last_centroid if tid not in seen_this_frame]
        for tid in stale:
            # Keep briefly? For MVP drop immediately when absent from frame.
            del self._last_centroid[tid]

        return events

    def snapshot_counts(self) -> dict[tuple[str, str, str], int]:
        """Return current window counts keyed by (lane, direction, class_name)."""
        out: dict[tuple[str, str, str], int] = {}
        for lane in self.lanes:
            for class_name, count in lane.counts.items():
                out[(lane.name, lane.direction, class_name)] = count
        return out
