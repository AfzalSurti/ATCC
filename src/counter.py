"""Per-movement line-crossing counters with track_id deduplication.

Supports junction models (2 / 6 / 8 ways) via ``src.junction``, and legacy
flat ``lanes`` lists for backward compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import supervision as sv

from src.junction import junction_to_lane_configs
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
    """Runtime state for one movement counting line and dedupe set."""

    name: str
    direction: str
    arm: str
    flow: str
    label: str
    line_start: tuple[float, float]
    line_end: tuple[float, float]
    line_zone: sv.LineZone
    counted_track_ids: set[int] = field(default_factory=set)
    counts: dict[str, int] = field(default_factory=dict)
    display_count: int = 0


class LaneCounter:
    """Count vehicles per junction movement / class using line crossings."""

    def __init__(
        self,
        config: dict[str, Any],
        on_count: Callable[[CountEvent], None] | None = None,
    ) -> None:
        """Build counting lines from ``junction`` or legacy ``lanes``.

        Args:
            config: Full pipeline config.
            on_count: Optional callback invoked for each new count event.
        """
        self.on_count = on_count
        self.lanes: list[LaneState] = []
        self._last_centroid: dict[int, tuple[float, float]] = {}
        self.junction_type: str = str((config.get("junction") or {}).get("type", "legacy"))

        lane_cfgs = junction_to_lane_configs(config)
        if not lane_cfgs:
            raise ValueError("No junction movements / lanes configured")

        for lane_cfg in lane_cfgs:
            name = str(lane_cfg["name"])
            flow = str(lane_cfg.get("flow") or lane_cfg.get("direction") or "in")
            arm = str(lane_cfg.get("arm") or name)
            label = str(lane_cfg.get("label") or name)
            line = lane_cfg["line"]
            if len(line) != 2:
                raise ValueError(f"Movement {name}: line must be [[x1,y1],[x2,y2]]")

            start_xy = (float(line[0][0]), float(line[0][1]))
            end_xy = (float(line[1][0]), float(line[1][1]))
            start = sv.Point(start_xy[0], start_xy[1])
            end = sv.Point(end_xy[0], end_xy[1])
            line_zone = sv.LineZone(start=start, end=end)
            self.lanes.append(
                LaneState(
                    name=name,
                    direction=flow,
                    arm=arm,
                    flow=flow,
                    label=label,
                    line_start=start_xy,
                    line_end=end_xy,
                    line_zone=line_zone,
                )
            )
            logger.info(
                "Movement %s | arm=%s flow=%s line %s → %s",
                name,
                arm,
                flow,
                line[0],
                line[1],
            )

    def reset_window_counts(self) -> None:
        """Clear per-movement class tallies after an interval rollover."""
        for lane in self.lanes:
            lane.counts.clear()

    def update(self, detections: list[Detection]) -> list[CountEvent]:
        """Return newly counted crossing events for this frame."""
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
                    movement_id=lane.name,
                    arm=lane.arm,
                    flow=lane.flow,
                    class_name=det.class_name,
                    class_id=det.class_id,
                    lane=lane.name,
                    direction=lane.flow,
                )
                events.append(event)
                logger.debug(
                    "COUNT track=%s movement=%s arm=%s flow=%s class=%s",
                    tid,
                    lane.name,
                    lane.arm,
                    lane.flow,
                    det.class_name,
                )
                if self.on_count is not None:
                    self.on_count(event)

        stale = [tid for tid in self._last_centroid if tid not in seen_this_frame]
        for tid in stale:
            del self._last_centroid[tid]

        return events

    def snapshot_counts(self) -> dict[tuple[str, str, str], int]:
        """Return counts keyed by (movement_id, flow, class_name)."""
        out: dict[tuple[str, str, str], int] = {}
        for lane in self.lanes:
            for class_name, count in lane.counts.items():
                out[(lane.name, lane.flow, class_name)] = count
        return out

    def movement_totals(self) -> dict[str, int]:
        """Total vehicles per movement id (all classes)."""
        return {lane.name: lane.display_count for lane in self.lanes}

    def class_totals(self) -> dict[str, int]:
        """Total vehicles per class across all movements."""
        totals: dict[str, int] = {}
        for lane in self.lanes:
            for cls_name, n in lane.counts.items():
                totals[cls_name] = totals.get(cls_name, 0) + n
        return totals
