"""Shared dataclasses used across pipeline stages.

Kept free of heavy ML imports so unit tests can import counting/aggregation
without loading Ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    """Standardized detection object returned by the detector / tracker."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None = None


@dataclass
class CountEvent:
    """A single counted vehicle crossing one junction movement line.

    Attributes:
        track_id: Persistent ByteTrack id (dedupe key per movement).
        movement_id: Unique id e.g. ``north_in``.
        arm: Approach name e.g. ``north``.
        flow: ``in`` (entering junction) or ``out`` (leaving toward arm).
        class_name: Vehicle class (bicycle / car / truck / …).
        class_id: Numeric class id from the detector taxonomy.
        lane: Alias of ``movement_id`` (legacy).
        direction: Alias of ``flow`` (legacy).
    """

    track_id: int
    movement_id: str
    arm: str
    flow: str
    class_name: str
    class_id: int
    lane: str = ""
    direction: str = ""

    def __post_init__(self) -> None:
        if not self.lane:
            self.lane = self.movement_id
        if not self.direction:
            self.direction = self.flow
