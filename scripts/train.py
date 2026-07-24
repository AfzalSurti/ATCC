#!/usr/bin/env python3
"""Phase 2 stub — fine-tune YOLOv8 on India-specific ATCC vehicle classes.

IMPORTANT:
  Full 16-class India taxonomy (Two Wheeler, Auto Rickshaw, MAV, Bullock Cart,
  Cycle Rickshaw, etc.) does NOT exist as a ready public drop-in dataset that
  matches Kotai's class list. Phase 2 requires manual collection + labeling
  (Roboflow or CVAT), then fine-tuning. Do NOT generate fake synthetic labels
  and pretend a production model exists.

TODO(Phase 2):
  1. Collect roadside multi-lane video across lighting/weather conditions.
  2. Label with the full target taxonomy in Roboflow/CVAT.
  3. Export YOLO-format dataset; update config/classes.
  4. Fine-tune:  yolo detect train data=data.yaml model=yolov8s.pt epochs=100
  5. Point detection.model_path at the best.pt weights.
  6. Delete detector.map_coco_to_target and use native class IDs.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Stub entrypoint for Phase 2 training."""
    print(__doc__)
    print("\nExiting: training is not part of Phase 1 MVP.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
