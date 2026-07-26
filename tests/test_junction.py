"""Tests for junction movement expansion (2 / 6 / 8 ways)."""

from __future__ import annotations

import pytest

from src.junction import (
    describe_junction_types,
    expected_movements,
    junction_to_lane_configs,
    movements_from_junction,
    normalize_junction_type,
)


def test_normalize_and_expected_counts() -> None:
    assert normalize_junction_type("4-way") == "four_way"
    assert expected_movements("two_way") == 2
    assert expected_movements("three_way") == 6
    assert expected_movements("four_way") == 8


def test_two_way_expands_to_two_movements() -> None:
    m = movements_from_junction(
        {
            "type": "two_way",
            "arms": [
                {
                    "name": "corridor",
                    "in_line": [[0, 0], [100, 0]],
                    "out_line": [[0, 50], [100, 50]],
                }
            ],
        }
    )
    assert len(m) == 2
    assert {x["flow"] for x in m} == {"in", "out"}


def test_four_way_requires_eight_movements() -> None:
    arms = []
    for name in ("north", "east", "south", "west"):
        arms.append(
            {
                "name": name,
                "in_line": [[0, 0], [10, 0]],
                "out_line": [[0, 10], [10, 10]],
            }
        )
    m = movements_from_junction({"type": "four_way", "arms": arms})
    assert len(m) == 8
    lanes = junction_to_lane_configs({"junction": {"type": "four_way", "arms": arms}})
    assert len(lanes) == 8


def test_three_way_wrong_arm_count_raises() -> None:
    with pytest.raises(ValueError, match="exactly 3 arms"):
        movements_from_junction({"type": "three_way", "arms": []})


def test_describe_types() -> None:
    ids = {d["id"] for d in describe_junction_types()}
    assert ids == {"two_way", "three_way", "four_way"}
