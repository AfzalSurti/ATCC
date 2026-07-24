"""Per-lane line-crossing counters with track_id deduplication.

Uses supervision ``LineZone`` for each configured lane. A vehicle is counted
once when its track centroid crosses the line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import supervision as sv

from src.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class CountEvent:
    """A single counted vehicle crossing."""

    track_id: int
    lane: str
    direction: str
    class_name: str
    class_id: int


@dataclass
class LaneState:
    """Runtime state for one lane's LineZone and dedupe set."""

    name: str
    direction: str
    line_zone: sv.LineZone
    counted_track_ids: set[int] = field(default_factory=set)
    # class_name -> count within the current aggregator window (local mirror)
    counts: dict[str, int] = field(default_factory=dict)


class LaneCounter:
    """Count vehicles per lane / direction / class using line crossings."""

    def __init__(
        self,
        config: dict[str, Any],
        on_count: Callable[[CountEvent], None] | None = None,
    ) -> None:
        """Build LineZones from ``lanes`` in config.

        Args:
            config: Full pipeline config.
            on_count: Optional callback invoked for each new count event.
        """
        self.on_count = on_count
        self.lanes: list[LaneState] = []
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

            start = sv.Point(float(line[0][0]), float(line[0][1]))
            end = sv.Point(float(line[1][0]), float(line[1][1]))
            line_zone = sv.LineZone(start=start, end=end)
            self.lanes.append(LaneState(name=name, direction=direction, line_zone=line_zone))
            logger.info("Lane %s (%s) line %s → %s", name, direction, line[0], line[1])

    def reset_window_counts(self) -> None:
        """Clear per-lane count mirrors after an aggregator interval rollover.

        Deduped track IDs are retained so a vehicle already counted is not
        double-counted in a later interval if it loiters on the line.
        """
        for lane in self.lanes:
            lane.counts.clear()

    def update(self, detections: list[Detection]) -> list[CountEvent]:
        """Update all lane LineZones; return newly counted events.

        Args:
            detections: Tracked detections (must include ``track_id``).

        Returns:
            List of new ``CountEvent`` objects for this frame.
        """
        events: list[CountEvent] = []
        if not detections:
            # Still tick empty detections so LineZone state ages out correctly.
            empty = sv.Detections.empty()
            for lane in self.lanes:
                lane.line_zone.trigger(empty)
            return events

        xyxy = np.array([list(d.bbox) for d in detections], dtype=np.float32)
        tracker_id = np.array([d.track_id if d.track_id is not None else -1 for d in detections], dtype=int)
        class_id = np.array([d.class_id for d in detections], dtype=int)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
        )

        # Map track_id → Detection for class name lookup after trigger.
        by_track = {d.track_id: d for d in detections if d.track_id is not None}

        for lane in self.lanes:
            before_in = int(lane.line_zone.in_count)
            before_out = int(lane.line_zone.out_count)
            crossed_in, crossed_out = lane.line_zone.trigger(sv_dets)
            after_in = int(lane.line_zone.in_count)
            after_out = int(lane.line_zone.out_count)

            # Prefer mask-based crossed tracks when available.
            crossed_masks = []
            if isinstance(crossed_in, np.ndarray):
                crossed_masks.append(crossed_in)
            if isinstance(crossed_out, np.ndarray):
                crossed_masks.append(crossed_out)

            newly_crossed_ids: set[int] = set()
            if crossed_masks:
                combined = np.zeros(len(sv_dets), dtype=bool)
                for m in crossed_masks:
                    if m.shape[0] == combined.shape[0]:
                        combined |= m.astype(bool)
                for i, flag in enumerate(combined):
                    if flag and tracker_id[i] >= 0:
                        newly_crossed_ids.add(int(tracker_id[i]))
            elif (after_in + after_out) > (before_in + before_out):
                # Fallback: if masks unavailable, attribute to detections present this frame
                # that have not yet been counted on this lane (best-effort).
                for d in detections:
                    if d.track_id is not None and d.track_id not in lane.counted_track_ids:
                        newly_crossed_ids.add(d.track_id)

            for tid in newly_crossed_ids:
                if tid in lane.counted_track_ids:
                    continue
                det = by_track.get(tid)
                if det is None:
                    continue
                lane.counted_track_ids.add(tid)
                lane.counts[det.class_name] = lane.counts.get(det.class_name, 0) + 1
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

        return events

    def snapshot_counts(self) -> dict[tuple[str, str, str], int]:
        """Return current window counts keyed by (lane, direction, class_name)."""
        out: dict[tuple[str, str, str], int] = {}
        for lane in self.lanes:
            for class_name, count in lane.counts.items():
                out[(lane.name, lane.direction, class_name)] = count
        return out
