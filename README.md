# ATCC Clone — Automatic Traffic Counter & Classifier

Video-based vehicle detection, classification, tracking, per-lane counting, and Excel export.

**Phase 1 MVP (this repo):** pretrained YOLOv8 (COCO vehicle classes) + ByteTrack + line-crossing counts + interval Excel reports on recorded video files.

> **Not production India-class taxonomy yet.** Full Kotai-style classes (auto rickshaw, MAV, bullock cart, etc.) need a custom labeled dataset (Phase 2). Phase 1 maps COCO `bicycle / motorcycle / car / bus / truck` as placeholders.

## Architecture

```
Video file
  → FrameProcessor (ROI, optional CLAHE, frame sampling)
  → Detector (YOLOv8 wrapper)
  → Tracker (ByteTrack via supervision)
  → LaneCounter (LineZone, dedupe by track_id)
  → Aggregator (15-min buckets by default)
  → ExcelExporter (.xlsx)
```

Each stage is a separate module under `src/` so the detector can later be swapped for TensorRT without touching counting logic.

## Setup (Windows / PowerShell)

```powershell
cd D:\ATCC
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

NVIDIA GPU: ensure a matching CUDA build of PyTorch is installed if you want GPU inference (`detection.device: "0"`). CPU works with `detection.device: "cpu"`.

## Configure

1. Put a traffic video at `data/sample_videos/sample1.mp4` (or any path).
2. Edit `config/camera_config.yaml`:
   - `source.path` — video path
   - `lanes[].line` — counting line endpoints in **pixel coordinates** of the source video
   - `export.interval_minutes` — report window (default 15)
3. Optional richer lane example: `configs_examples/sample_lanes.yaml`

## Run on a video (Phase 1)

```powershell
python scripts/run_on_video.py --config config/camera_config.yaml --video data/sample_videos/sample1.mp4
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--model yolov8s.pt` | Larger COCO checkpoint |
| `--no-viz` | Skip annotated MP4 |
| `--log-level DEBUG` | Verbose timing / counts |

Outputs:

- Excel: `outputs/reports/atcc_report_YYYYMMDD.xlsx`
- Annotated video (if enabled): `outputs/annotated/annotated_*.mp4`

## Tests

```powershell
pytest tests/ -q
```

## Phase roadmap

| Phase | Status | Scope |
|-------|--------|--------|
| 1 | **Implemented** | File video, COCO classes, ByteTrack, Excel |
| 2 | Stub `scripts/train.py` | Custom India classes + fine-tune |
| 3 | Stub `scripts/run_on_stream.py` | RTSP + Streamlit dashboard |
| 4 | Stub `TensorRTDetector` | ONNX/TensorRT edge deploy |

See **[docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md)** for full technical detail of every module, data flow, class mapping, and why each design choice was made.

## Reference product (feature parity target)

[Kotai Electronics ATCC](https://kotaielectronics.com/automatic-traffic-counter-and-classifier/)
