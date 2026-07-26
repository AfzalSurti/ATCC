"""Junction geometry helpers: 2-way / 3-way / 4-way → IN/OUT movements.

Model used by ATCC:

* **two_way**  — 1 road, 2 movements (incoming + outgoing) → **2 ways**
* **three_way** — T-junction, 3 arms × (IN + OUT) → **6 ways**
* **four_way**  — crossroads, 4 arms × (IN + OUT) → **8 ways**

IN  = vehicles entering the junction from that arm.
OUT = vehicles leaving the junction toward that arm.

Each movement has its own counting line in the camera frame. Counts are also
broken down by vehicle class (bicycle, motorcycle, car, bus, truck).
"""

from __future__ import annotations

from typing import Any

JUNCTION_MOVEMENT_COUNTS: dict[str, int] = {
    "two_way": 2,
    "three_way": 6,
    "four_way": 8,
}

DEFAULT_ARM_NAMES: dict[str, list[str]] = {
    "two_way": ["north", "south"],  # one corridor; each end is an arm with in+out → 4?
}


# User said 2-way = only 2 ways incoming and outgoing.
# So for two_way we use a single corridor with two movements: inbound + outbound
# (not 2 arms × 2). Naming: approach_in / approach_out OR northbound / southbound.

def expected_movements(junction_type: str) -> int:
    """Return how many counting movements a junction type must define."""
    key = normalize_junction_type(junction_type)
    return JUNCTION_MOVEMENT_COUNTS[key]


def normalize_junction_type(junction_type: str) -> str:
    """Normalize aliases to two_way | three_way | four_way."""
    raw = str(junction_type or "four_way").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "2": "two_way",
        "2_way": "two_way",
        "two": "two_way",
        "twoway": "two_way",
        "3": "three_way",
        "3_way": "three_way",
        "three": "three_way",
        "threeway": "three_way",
        "t_junction": "three_way",
        "4": "four_way",
        "4_way": "four_way",
        "four": "four_way",
        "fourway": "four_way",
        "crossroads": "four_way",
    }
    key = aliases.get(raw, raw)
    if key not in JUNCTION_MOVEMENT_COUNTS:
        raise ValueError(
            f"Unknown junction type {junction_type!r}. "
            f"Use one of: {sorted(JUNCTION_MOVEMENT_COUNTS)}"
        )
    return key


def default_arm_names(junction_type: str) -> list[str]:
    """Default geographic arm labels for a junction type."""
    key = normalize_junction_type(junction_type)
    if key == "two_way":
        # Single corridor: two directional movements (not 4).
        return ["corridor"]
    if key == "three_way":
        return ["north", "east", "west"]  # classic T opening south omitted / or N/E/S
    return ["north", "east", "south", "west"]


def movements_from_junction(junction_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a ``junction`` config block into flat movement definitions.

    Returns list of dicts with keys:
      movement_id, arm, flow (in|out), label, line [[x,y],[x,y]]
    """
    jtype = normalize_junction_type(str(junction_cfg.get("type", "four_way")))
    arms = list(junction_cfg.get("arms") or [])

    movements: list[dict[str, Any]] = []

    if jtype == "two_way":
        # Prefer explicit movements list; else one arm with in+out; else legacy two lines.
        explicit = list(junction_cfg.get("movements") or [])
        if explicit:
            for m in explicit:
                flow = str(m.get("flow", m.get("direction", "in"))).lower()
                arm = str(m.get("arm", "corridor"))
                mid = str(m.get("id") or m.get("name") or f"{arm}_{flow}")
                movements.append(
                    {
                        "movement_id": mid,
                        "arm": arm,
                        "flow": flow if flow in {"in", "out"} else "in",
                        "label": str(m.get("label") or mid),
                        "line": m["line"],
                    }
                )
        elif arms:
            # If user provided 1 arm with in/out → 2 movements.
            # If 2 arms each with only one line, treat as opposite directions.
            if len(arms) == 1:
                arm = arms[0]
                name = str(arm.get("name", "corridor"))
                for flow, key in (("in", "in_line"), ("out", "out_line")):
                    line = arm.get(key) or arm.get("line")
                    if not line:
                        raise ValueError(f"two_way arm {name} missing {key}")
                    movements.append(
                        {
                            "movement_id": f"{name}_{flow}",
                            "arm": name,
                            "flow": flow,
                            "label": f"{name} {flow.upper()}",
                            "line": line,
                        }
                    )
            else:
                # two arms → each contributes its primary travel direction as one movement
                # Map first arm → out-bound style "in" toward junction, second similarly.
                for arm in arms[:2]:
                    name = str(arm.get("name", "arm"))
                    flow = str(arm.get("flow", "in")).lower()
                    if flow not in {"in", "out"}:
                        flow = "in"
                    line = arm.get("in_line") or arm.get("out_line") or arm.get("line")
                    if not line:
                        raise ValueError(f"two_way arm {name} needs a line")
                    movements.append(
                        {
                            "movement_id": f"{name}_{flow}",
                            "arm": name,
                            "flow": flow,
                            "label": str(arm.get("label") or f"{name} {flow.upper()}"),
                            "line": line,
                        }
                    )
        else:
            raise ValueError("two_way junction requires arms or movements with lines")

    else:
        # three_way / four_way: each arm must define in_line and out_line
        expected_arms = 3 if jtype == "three_way" else 4
        if len(arms) != expected_arms:
            raise ValueError(
                f"{jtype} requires exactly {expected_arms} arms (got {len(arms)}). "
                f"Each arm needs in_line + out_line → {expected_arms * 2} movements."
            )
        for arm in arms:
            name = str(arm["name"])
            for flow, key in (("in", "in_line"), ("out", "out_line")):
                line = arm.get(key)
                if not line or len(line) != 2:
                    raise ValueError(f"Arm {name}: {key} must be [[x1,y1],[x2,y2]]")
                movements.append(
                    {
                        "movement_id": f"{name}_{flow}",
                        "arm": name,
                        "flow": flow,
                        "label": str(arm.get("label") or f"{name} {flow.upper()}"),
                        "line": line,
                    }
                )

    expected = expected_movements(jtype)
    if len(movements) != expected:
        raise ValueError(
            f"{jtype} must define {expected} movements (got {len(movements)})."
        )
    return movements


def junction_to_lane_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert junction block (or legacy lanes) into counter lane configs.

    Prefer ``junction`` when present; otherwise keep legacy ``lanes``.
    """
    if config.get("junction"):
        movements = movements_from_junction(config["junction"])
        return [
            {
                "name": m["movement_id"],
                "direction": m["flow"],
                "arm": m["arm"],
                "flow": m["flow"],
                "label": m["label"],
                "line": m["line"],
            }
            for m in movements
        ]
    return list(config.get("lanes") or [])


def apply_junction_type(config: dict[str, Any], junction_type: str) -> dict[str, Any]:
    """Set ``junction.type`` on a config copy and validate movement count."""
    cfg = dict(config)
    j = dict(cfg.get("junction") or {})
    j["type"] = normalize_junction_type(junction_type)
    cfg["junction"] = j
    # Validate early
    junction_to_lane_configs(cfg)
    return cfg


def describe_junction_types() -> list[dict[str, Any]]:
    """API/UI metadata for junction types."""
    return [
        {
            "id": "two_way",
            "label": "2-way road",
            "arms": 1,
            "movements": 2,
            "description": "One corridor: incoming + outgoing (2 ways).",
        },
        {
            "id": "three_way",
            "label": "3-way / T-junction",
            "arms": 3,
            "movements": 6,
            "description": "Three approaches, each with IN + OUT (6 ways).",
        },
        {
            "id": "four_way",
            "label": "4-way crossroads",
            "arms": 4,
            "movements": 8,
            "description": "Four approaches, each with IN + OUT (8 ways).",
        },
    ]
