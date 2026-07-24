"""Unit tests for LaneCounter track_id deduplication and line crossing."""

from __future__ import annotations

from src.counter import LaneCounter
from src.schemas import Detection


def _config() -> dict:
    return {
        "classes": {"target": ["car", "bus", "truck", "motorcycle", "bicycle"]},
        "lanes": [
            {
                "name": "lane_1",
                "direction": "north",
                # Horizontal line across mid frame
                "line": [[0, 100], [200, 100]],
            }
        ],
    }


def _det(track_id: int, cx: float, cy: float, class_name: str = "car") -> Detection:
    w = 20.0
    h = 20.0
    return Detection(
        bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        track_id=track_id,
    )


def test_same_track_counted_once_when_crossing() -> None:
    """A single track crossing the line must produce exactly one count event."""
    counter = LaneCounter(_config())
    # Approach from above the line, then cross below.
    events_1 = counter.update([_det(1, 100, 60)])
    events_2 = counter.update([_det(1, 100, 140)])
    all_events = events_1 + events_2
    assert len(all_events) == 1
    assert all_events[0].track_id == 1
    assert all_events[0].lane == "lane_1"
    assert all_events[0].class_name == "car"

    # Same track lingering / oscillating must not recount.
    events_3 = counter.update([_det(1, 100, 150)])
    events_4 = counter.update([_det(1, 100, 90)])
    events_5 = counter.update([_det(1, 100, 160)])
    assert events_3 + events_4 + events_5 == []


def test_different_tracks_counted_separately() -> None:
    """Two distinct track IDs crossing should produce two events."""
    counter = LaneCounter(_config())
    counter.update([_det(10, 80, 50)])
    e1 = counter.update([_det(10, 80, 150)])
    counter.update([_det(11, 120, 50)])
    e2 = counter.update([_det(11, 120, 150)])
    assert len(e1) == 1
    assert len(e2) == 1
    assert {e1[0].track_id, e2[0].track_id} == {10, 11}


def test_snapshot_counts_by_class() -> None:
    """Per-lane class tallies should reflect counted events."""
    counter = LaneCounter(_config())
    counter.update([_det(1, 100, 50, "car")])
    counter.update([_det(1, 100, 150, "car")])
    counter.update([_det(2, 100, 50, "bus")])
    counter.update([_det(2, 100, 150, "bus")])
    snap = counter.snapshot_counts()
    assert snap[("lane_1", "north", "car")] == 1
    assert snap[("lane_1", "north", "bus")] == 1
