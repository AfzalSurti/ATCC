# ATCC — Automatic Traffic Counter & Classifier

Video-based vehicle detection, classification, tracking, per-lane counting, Excel export, live RTSP, Streamlit dashboard, and ONNX/TensorRT export paths.

| Phase | Status | What |
|-------|--------|------|
| 1 | Done | File video + COCO YOLOv8 + ByteTrack + Excel |
| 2 | Done (needs your labels) | `scripts/train.py` + India class taxonomy + `classes.mode: custom` |
| 3 | Done | RTSP/webcam live source + Streamlit dashboard |
| 4 | Done | ONNX / TensorRT export + `detection.backend` swap |

> **India 16-class accuracy** still needs real labeled data (Roboflow/CVAT). Phase 2 provides the training path — it does not invent synthetic labels.

## Architecture

```
Video (file / RTSP / webcam)
  → Preprocess (ROI, CLAHE, sampling)
  → Detector (ultralytics | onnx | tensorrt)
  → Tracker (ByteTrack)
  → LaneCounter (line crossing, track_id dedupe)
  → Aggregator (15-min buckets)
  → ExcelExporter
  → StatsStore → Streamlit dashboard
```

## Setup

```powershell
cd D:\ATCC
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Phase 1 — recorded video

1. Put a video at `data/sample_videos/sample1.mp4`
2. Calibrate `lanes` in `config/camera_config.yaml`
3. Run:

```powershell
python scripts/run_on_video.py --video data/sample_videos/sample1.mp4
```

Excel → `outputs/reports/` · annotated MP4 → `outputs/annotated/`

## Phase 2 — custom India classes (requires your dataset)

1. Label with classes from `config/india_classes.yaml`
2. Export YOLO format under `data/datasets/<name>/` (see `config/dataset.example.yaml`)
3. Train:

```powershell
python scripts/train.py --data data/datasets/<name>/data.yaml --model yolov8s.pt --epochs 100
```

4. Point config at `best.pt` and set `classes.mode: custom` (target list = data.yaml order)

## Phase 3 — live stream + dashboard

```powershell
# RTSP or webcam index
python scripts/run_on_stream.py --rtsp rtsp://user:pass@cam/stream1
python scripts/run_on_stream.py --rtsp 0

# Streamlit live view + stats
python scripts/run_dashboard.py --config config/camera_config.yaml --source rtsp://...
```

## Phase 4 — edge export

```powershell
python scripts/export_onnx.py --weights yolov8n.pt --outdir outputs/models
# optional TensorRT (CUDA + TensorRT required):
python scripts/export_onnx.py --weights yolov8n.pt --tensorrt
```

Then in config:

```yaml
detection:
  backend: onnx   # or tensorrt
  model_path: "outputs/models/yolov8n.onnx"
```

## Tests

```powershell
pytest tests/ -q
```

## Docs

Full technical detail: **[docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md)**

## Reference

[Kotai Electronics ATCC](https://kotaielectronics.com/automatic-traffic-counter-and-classifier/)
