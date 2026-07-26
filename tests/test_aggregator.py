"""Unit tests for time-bucketed Aggregator rollover and flush."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.aggregator import Aggregator, IntervalBucket
from src.schemas import CountEvent


def _event(class_name: str = "car", movement_id: str = "north_in") -> CountEvent:
    arm, flow = movement_id.rsplit("_", 1)
    return CountEvent(
        track_id=1,
        movement_id=movement_id,
        arm=arm,
        flow=flow,
        class_name=class_name,
        class_id=0,
    )


def test_interval_rollover_invokes_callback() -> None:
    """Crossing the interval boundary should complete the prior bucket."""
    completed: list[IntervalBucket] = []
    start = datetime(2026, 7, 24, 10, 0, 0, tzinfo=timezone.utc)
    agg = Aggregator(
        {"export": {"interval_minutes": 15}},
        on_interval_complete=completed.append,
        session_start=start,
    )

    agg.add_event(_event("car"), start + timedelta(minutes=1))
    agg.add_event(_event("bus"), start + timedelta(minutes=14))
    assert completed == []

    agg.observe_time(start + timedelta(minutes=15, seconds=1))
    assert len(completed) == 1
    assert completed[0].counts[("north_in", "in", "car")] == 1
    assert completed[0].counts[("north_in", "in", "bus")] == 1
    assert completed[0].interval_start == start
    assert completed[0].interval_end == start + timedelta(minutes=15)


def test_flush_exports_partial_window() -> None:
    """Graceful shutdown must flush the in-progress interval."""
    completed: list[IntervalBucket] = []
    start = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
    agg = Aggregator(
        {"export": {"interval_minutes": 15}},
        on_interval_complete=completed.append,
        session_start=start,
    )
    agg.add_event(_event("truck"), start + timedelta(minutes=3))
    flushed = agg.flush()
    assert flushed is not None
    assert len(completed) == 1
    assert completed[0].counts[("north_in", "in", "truck")] == 1
    assert agg.flush() is None
