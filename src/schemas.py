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
    """A single counted vehicle crossing a lane line."""

    track_id: int
    lane: str
    direction: str
    class_name: str
    class_id: int
