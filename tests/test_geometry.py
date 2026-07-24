"""Tests for centroid line-segment intersection used by LaneCounter."""

from __future__ import annotations

from src.counter import _segments_intersect


def test_segments_intersect_crossing() -> None:
    """Vertical path across a horizontal line should intersect."""
    assert _segments_intersect((100, 50), (100, 150), (0, 100), (200, 100))


def test_segments_no_intersect_parallel() -> None:
    """Two parallel horizontal segments should not intersect."""
    assert not _segments_intersect((0, 50), (100, 50), (0, 100), (100, 100))
