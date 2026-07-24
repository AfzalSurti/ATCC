# ATCC Clone — Technical Design Document

This document explains **everything built in Phase 1**: architecture, each module, data contracts, config knobs, limitations, and how later phases plug in. Read this if you want to understand *why* the code is structured the way it is—not just how to run it.

---

## 1. Problem statement

An **Automatic Traffic Counter and Classifier (ATCC)** ingests traffic video (file now; RTSP later), detects vehicles, classifies them, tracks them across frames, counts them once per lane/direction when they cross a virtual line, aggregates counts into time windows (e.g. every 15 minutes), and writes Excel reports.

We are targeting feature parity with products like [Kotai ATCC](https://kotaielectronics.com/automatic-traffic-counter-and-classifier/), but **Phase 1 is an MVP architecture proof** on a normal PC (preferably NVIDIA CUDA). It does **not** claim accurate India-specific 16-class classification yet.

---

## 2. What Phase 1 deliberately does and does not do

### Does

- End-to-end pipeline on a **recorded video file**
- **YOLOv8** pretrained on **COCO** (5 vehicle-ish classes used)
- **ByteTrack** tracking via Roboflow `supervision` (not a hand-rolled tracker)
- Per-lane **line-crossing** counts with **track_id deduplication**
- Time-bucketed aggregation (default **15 minutes**)
- Excel export (`.xlsx`) via pandas + openpyxl
- YAML-driven config (no hardcoded paths in logic)
- Modular stages so detector backend can be swapped later
- Unit tests for counter dedupe + aggregator rollover
- Stubs/TODOs for Phases 2–4

### Does not

- Background subtraction (MOG2/KNN) as detection — **forbidden by design**
- Fake synthetic India-class training data
- Pretending a fine-tuned custom model exists
- Full RTSP live pipeline (Phase 3 stub only)
- Streamlit dashboard (Phase 3)
- TensorRT / Jetson deployment (Phase 4 stub only)

---

## 3. High-level pipeline

```
┌─────────────┐   FramePacket    ┌────────────────┐
│ VideoSource │ ───────────────► │ Preprocessor   │  ROI / CLAHE / motion-skip gate
└─────────────┘                  └───────┬────────┘
                                         │ BGR frame
                                         ▼
                                 ┌────────────────┐
                                 │ Detector (YOLO)│  boxes + COCO→target class map
                                 └───────┬────────┘
                                         │ list[Detection]
                                         ▼
                                 ┌────────────────┐
                                 │ Tracker        │  ByteTrack → track_id
                                 └───────┬────────┘
                                         │ tracked Detection
                                         ▼
                                 ┌────────────────┐
                                 │ LaneCounter    │  LineZone cross + dedupe
                                 └───────┬────────┘
                                         │ CountEvent
                                         ▼
                                 ┌────────────────┐
                                 │ Aggregator     │  15-min buckets
                                 └───────┬────────┘
                                         │ IntervalBucket (on rollover / flush)
                                         ▼
                                 ┌────────────────┐
                                 │ ExcelExporter  │  outputs/reports/*.xlsx
                                 └────────────────┘
```

Orchestration lives in `src/pipeline.py` (`Pipeline.run()`).

---

## 4. Repository layout

| Path | Role |
|------|------|
| `config/camera_config.yaml` | Single runtime source of truth |
| `configs_examples/sample_lanes.yaml` | Example multi-lane geometry |
| `src/config_loader.py` | YAML load + path resolution vs project root |
| `src/schemas.py` | Shared `Detection` / `CountEvent` dataclasses (no heavy ML imports) |
| `src/video_source.py` | File capture + FPS sampling → `FramePacket` |
| `src/preprocessing.py` | ROI mask, CLAHE, optional motion skip |
| `src/detector.py` | Ultralytics YOLO wrapper + COCO mapping |
| `src/tracker.py` | `supervision.ByteTrack` wrapper |
| `src/counter.py` | Per-lane `LineZone`, dedupe, `CountEvent` |
| `src/aggregator.py` | Time windows → `IntervalBucket` |
| `src/exporter.py` | Excel writer (append / session modes) |
| `src/pipeline.py` | Wire stages, timing logs, graceful flush |
| `scripts/run_on_video.py` | Phase 1 CLI |
| `scripts/run_on_stream.py` | Phase 3 stub |
| `scripts/train.py` | Phase 2 stub |
| `tests/` | Counter + aggregator unit tests |
| `data/sample_videos/` | Local videos (gitignored) |
| `outputs/reports/` | Generated Excel (gitignored) |
| `outputs/annotated/` | Debug MP4 overlays (gitignored) |

---

## 5. Core data contracts

### `FramePacket` (`video_source.py`)

| Field | Meaning |
|-------|---------|
| `frame` | BGR `numpy` image |
| `frame_index` | Index among **sampled** frames |
| `source_frame_index` | Index in the raw file |
| `timestamp` | UTC datetime (file: session_start + index/fps) |
| `source_fps` | Native video FPS |

### `Detection` (`detector.py`)

| Field | Meaning |
|-------|---------|
| `bbox` | `(x1, y1, x2, y2)` in **source pixel** space |
| `class_id` | Target taxonomy id (Phase 1 mapped) |
| `class_name` | Report class string |
| `confidence` | Detector score |
| `track_id` | Set by tracker; `None` before tracking |

### `CountEvent` (`counter.py`)

Emitted once per `(track_id, lane)` when the track centroid crosses that lane’s line.

### `IntervalBucket` (`aggregator.py`)

Holds `interval_start`, `interval_end`, and `counts[(lane, direction, class_name)] → int`.

Keeping these as plain dataclasses (not framework-specific types) makes it easy to swap YOLO for TensorRT later: only `Detector.predict()` must keep returning `list[Detection]`.

---

## 6. Module deep-dive

### 6.1 Video source — why sample FPS?

Traffic cameras often run 25–30 FPS. Running YOLO on every frame is expensive and usually unnecessary for counting. Config `frame_sampling.target_fps` (default 12) derives a stride:

```text
stride = round(source_fps / target_fps)
```

Only every Nth frame is processed. Timestamps for files are derived from FPS + frame index (anchored to session start so Excel intervals look like a real session). Live streams will use wall-clock time in Phase 3 (`LiveVideoSource` stub raises `NotImplementedError` today).

### 6.2 Preprocessing — ROI, CLAHE, motion skip

- **ROI polygon**: pixels outside the road are zeroed via a mask so off-road detections (sidewalk pedestrians misread as bikes, parked lots, etc.) are reduced. Geometry for counting lines stays in full-frame coordinates; we mask the image fed to YOLO.
- **CLAHE**: optional lighting normalization under harsh sun/shadow.
- **Motion skip**: optional gate comparing consecutive grayscale frames; if almost nothing moved, skip inference. This is **not** MOG2/KNN detection—it only decides whether to call YOLO.

**Resize/letterbox:** Ultralytics letterboxes internally using `preprocessing.imgsz`. We intentionally keep the working frame at source resolution so LineZone coordinates match the video you calibrated.

### 6.3 Detector — COCO placeholder mapping

Phase 1 loads `yolov8n.pt` (or whatever `detection.model_path` says) and filters to COCO vehicle IDs: bicycle(1), car(2), motorcycle(3), bus(5), truck(7).

**Critical honesty:** COCO “truck” ≠ Indian “2-axle rigid” vs “MAV” vs “articulated”. COCO has no auto rickshaw / bullock cart / cycle rickshaw. Therefore:

1. Config lists target class names under `classes.target` (single source of truth).
2. `map_coco_to_target()` in `detector.py` is a **clearly labeled placeholder**.
3. Phase 2 deletes that function and points `model_path` at fine-tuned weights.

`TensorRTDetector` is a stub with the same intended interface for Phase 4.

### 6.4 Tracker — why ByteTrack via supervision?

Counting must be **identity-stable**. Without tracking, the same car overlapping the line for 10 frames would count 10 times. ByteTrack assigns a persistent `track_id`. We wrap `supervision.ByteTrack` so we do not maintain our own Kalman/IoU association code.

### 6.5 Counter — LineZone + dedupe

Each lane in YAML:

```yaml
- name: "lane_1_northbound"
  direction: "north"
  line: [[x1,y1],[x2,y2]]
```

Counting uses per-track **centroid path × counting-line segment intersection**,
with `supervision.LineZone` objects retained for annotated-video display.
(`LineZone.trigger` is avoided as the sole counting path because it breaks on
NumPy 2.x `np.cross` changes; requirements pin `numpy==1.26.4`.)


Dedup set survives interval rollover (a vehicle already counted must not be counted again in the next 15-minute sheet if it still intersects the line). Per-lane class tallies for the *current window* are cleared on rollover via `reset_window_counts()`.

**Calibration tip:** Open a frame in an image editor, note pixel coords for each lane’s counting line, paste into YAML. Lines should cut the lane perpendicular to traffic flow where vehicles are clearly separated.

### 6.6 Aggregator — 15-minute windows

Default `export.interval_minutes: 15` matches typical ATCC report cadence. First window starts at session start. When frame time ≥ `interval_end`, the bucket is completed, handed to the exporter callback, and a new bucket opens. On Ctrl+C / normal end, `flush()` writes the **partial** final window so counts are not lost.

### 6.7 Excel exporter

Columns:

`Interval Start | Interval End | Lane | Direction | Vehicle Class | Count`

Modes:

- `append` (default): one workbook per calendar day, rows appended per interval
- `session`: new file per run (`atcc_report_<timestamp>.xlsx`)

Sheet strategy:

- `single`: one sheet named `counts`
- `per_day`: sheet name = `YYYY-MM-DD`

Empty intervals still write a zero marker row so gaps are visible in audits.

### 6.8 Pipeline orchestration

`Pipeline.run()`:

1. Opens video, builds ROI mask from frame size
2. For each sampled frame: preprocess → detect → track → count → aggregate
3. Logs per-stage milliseconds (`logging.per_frame_timing`)
4. Optionally writes annotated MP4 (boxes, IDs, LineZones, total counts)
5. `finally`: flush aggregator + release writer

This is the only place that knows the full graph—stages stay independently testable.

---

## 7. Configuration reference

See `config/camera_config.yaml` for the live template. Important groups:

| Key | Purpose |
|-----|---------|
| `source.type/path` | `file` + path (RTSP blocked until Phase 3) |
| `frame_sampling.target_fps` | Inference rate target |
| `roi.enabled/polygon` | Road area mask |
| `lanes[]` | Names, directions, counting lines |
| `detection.*` | Model path, conf, IoU, device |
| `classes.target` / `coco_vehicle_ids` | Class taxonomy source of truth |
| `tracking.*` | ByteTrack hyperparameters |
| `export.*` | Interval, output dir, mode, sheets |
| `visualization.*` | Annotated video toggle |
| `logging.*` | Level + timing |

---

## 8. India class list vs Phase 1 reality

Reference product classes (India-oriented), for future Phase 2 labeling:

Two Wheeler, Three Wheeler/Auto Rickshaw, Car/Jeep/Van/Taxi, Mini Bus, Standard Bus, 2-Axle Rigid Truck, MAV Rigid Truck, Articulated/Semi-Articulated Truck, Bullock Cart, Horse Drawn, LCV Freight, Cycle, Cycle Rickshaw, 3-Axle Rigid Truck, …

**Phase 1 report classes:** `bicycle`, `motorcycle`, `car`, `bus`, `truck` (COCO-mapped).

Do not expand the YAML class list to 16 names without a model that can actually predict them—that would produce misleading Excel “zeros” and false confidence.

---

## 9. Performance notes (profiling hooks)

Pipeline logs look like:

```text
frame 42 | det=12.3ms track=1.1ms count=0.4ms total=15.0ms | dets=5 tracks=5 events=0
```

Typical bottleneck: **detector**. Mitigations already available:

- Lower `target_fps`
- Smaller model (`yolov8n` vs `yolov8s`)
- Smaller `imgsz`
- Enable `motion_skip`
- Use CUDA (`device: "0"`)

Counting/tracking are cheap relative to YOLO on GPU or CPU.

---

## 10. Testing strategy

| Test | Guarantees |
|------|------------|
| `tests/test_counter.py` | Same `track_id` counted once; different IDs separate; class snapshot |
| `tests/test_aggregator.py` | 15-min rollover callback; flush partial window |

These tests **do not** download YOLO weights—they unit-test counting math with synthetic boxes.

---

## 11. Phase 2–4 plug-in plan

### Phase 2 — Custom train (`scripts/train.py` stub)

1. Collect video across sites/lighting
2. Label full taxonomy (Roboflow/CVAT)
3. `yolo detect train ...`
4. Set `detection.model_path` → `best.pt`
5. Remove `map_coco_to_target`; update `classes.target`

### Phase 3 — Live + dashboard

- Implement `LiveVideoSource` (RTSP reconnect, wall-clock timestamps)
- Flesh out `scripts/run_on_stream.py`
- Add Streamlit live view consuming shared stats (not built in Phase 1)

### Phase 4 — Edge

- Export ONNX → TensorRT engine
- Implement `TensorRTDetector.predict()` returning the same `Detection` list
- No changes required to tracker/counter/aggregator/exporter if the contract holds

---

## 12. What was built in this Cursor session (task log)

Commits were made per logical task and pushed to `origin`:

1. **Scaffold** — folders, `.gitignore`, `requirements.txt`, YAML configs, placeholders
2. **Ingest + preprocess + detect + track** — modular stage implementations
3. **Count + aggregate + export + pipeline** — full Phase 1 orchestration
4. **CLI + tests + Phase stubs** — `run_on_video.py`, pytest, Phase 2–4 stubs
5. **Docs** — `README.md` + this technical design document

---

## 13. How to extend safely

- **New lane:** add YAML entry only; restart run.
- **New export column:** change `exporter.COLUMNS` + `_bucket_to_rows`.
- **New detector backend:** subclass / replace `Detector`, keep `predict(frame) -> list[Detection]`.
- **Custom classes:** only after real labels + fine-tune (Phase 2).

---

*Document version: Phase 1 MVP — aligned with package version `0.1.0`.*
