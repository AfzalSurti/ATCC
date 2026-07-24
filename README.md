# ATCC — Automatic Traffic Counter & Classifier

Video-based vehicle detection, classification, tracking, per-lane counting, Excel export, **React upload dashboard**, live RTSP, Streamlit (optional), and ONNX/TensorRT export.

| Phase | Status | What |
|-------|--------|------|
| 1 | Done | File video + COCO YOLOv8 + ByteTrack + Excel |
| 2 | Done (needs your labels) | `scripts/train.py` + India class taxonomy |
| 3 | Done | RTSP/webcam + Streamlit + **React upload UI** |
| 4 | Done | ONNX / TensorRT export |

## React dashboard (upload videos in the browser)

Users upload **one or many** videos from the UI. Files go to the API — you do **not** copy them into a backend folder by hand.

```powershell
# Terminal 1 — API
cd D:\ATCC
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_api.py

# Terminal 2 — React
cd D:\ATCC\frontend
npm install
npm run dev
```

Open **http://localhost:5173** → drop/select videos → Process → download Excel when each video completes.

API docs: http://127.0.0.1:8000/docs

## Architecture

```
Browser (React)
  → POST /api/upload (multipart, 1..N videos)
  → FastAPI job worker
  → ATCC Pipeline (detect → track → count → Excel)
  → GET /api/jobs/{id} + Excel download links
```

## Setup (Python)

```powershell
cd D:\ATCC
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## CLI (optional)

```powershell
python scripts/run_on_video.py --video path\to\video.mp4
python scripts/run_on_stream.py --rtsp 0
python scripts/run_dashboard.py   # Streamlit (legacy live view)
```

Calibrate counting lines in `config/camera_config.yaml` before serious runs.

## Docs

See [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md).
