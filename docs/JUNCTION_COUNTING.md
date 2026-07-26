# Junction counting model (2-way / 3-way / 4-way)

This document explains **how ATCC counts traffic at intersections**, what “8 ways / 6 ways / 2 ways” means, which software pieces are used, and how to calibrate lines for your camera.

---

## 1. What we count

For each **movement (way)** the system records:

| Field | Meaning |
|--------|---------|
| **Arm** | Approach name (north / east / south / west, or corridor) |
| **Flow** | `IN` = entering the junction from that arm · `OUT` = leaving toward that arm |
| **Movement id** | `{arm}_{in\|out}` e.g. `north_in` |
| **Vehicle class** | From the detector: `bicycle`, `motorcycle`, `car`, `bus`, `truck` |
| **Count** | Number of unique tracked vehicles that crossed that line |

Excel columns:

`Interval Start | Interval End | Junction Type | Arm | Flow | Movement | Vehicle Class | Count`

---

## 2. Junction types → number of ways

| Type | UI name | Arms | Movements (ways) | Rule |
|------|---------|------|------------------|------|
| `two_way` | 2-way road | 1 corridor | **2** | Incoming + outgoing only |
| `three_way` | 3-way / T-junction | 3 | **6** | Each arm has IN + OUT |
| `four_way` | 4-way crossroads | 4 | **8** | Each arm has IN + OUT |

```
                 OUT ↑        IN ↓
                    │  NORTH  │
                    │         │
        IN →  WEST ─┼─────────┼─ EAST  ← IN
                    │         │
                    │  SOUTH  │
                 IN ↑        OUT ↓
```

Example **4-way (8 ways):**  
`north_in`, `north_out`, `east_in`, `east_out`, `south_in`, `south_out`, `west_in`, `west_out`.

This is **approach-based volume counting** (how many vehicles enter/leave each arm), not a full 12-cell turning-movement matrix (left/through/right from every arm). Turning OD matrices can be added later with entry→exit track pairing.

---

## 3. What technology we use (stack)

| Stage | Library / module | Role |
|-------|------------------|------|
| Video upload UI | **React + Vite** (`frontend/`) | User picks junction type and uploads 1..N videos |
| API | **FastAPI** (`api/`) | Receives files + `junction_type`, runs jobs |
| Frame I/O | **OpenCV** (`src/video_source.py`) | Read uploaded file frames |
| Detection | **Ultralytics YOLOv8** (`src/detector.py`) | Boxes + class (COCO vehicles for Phase 1) |
| Tracking | **supervision ByteTrack** (`src/tracker.py`) | Persistent `track_id` so each vehicle counts once |
| Geometry | **`src/junction.py`** | Expands 2/3/4-way config → 2/6/8 counting lines |
| Counting | **`src/counter.py`** | Centroid path × line segment intersection + track dedupe |
| Aggregation | **`src/aggregator.py`** | Buckets by time interval (default 15 min) |
| Export | **pandas + openpyxl** (`src/exporter.py`) | `.xlsx` per session/interval |
| Config | **YAML** (`config/camera_config.yaml`) | Arm `in_line` / `out_line` pixel coordinates |

**Not used for detection:** background subtraction (MOG2/KNN). Detection is always YOLO.

---

## 4. How a count happens (algorithm)

1. YOLO detects vehicles each sampled frame (~12 FPS target).
2. ByteTrack assigns a stable `track_id`.
3. For each track, we keep the previous bbox **centroid**.
4. If the segment *previous centroid → current centroid* intersects a movement’s counting **line**, and that `track_id` was not counted on that movement before → **+1** for that movement + class.
5. Totals are written to Excel (and shown in the React job card).

This is why calibration of lines matters: lines must cut the approach where vehicles clearly cross when entering or leaving.

---

## 5. Vehicle classes

Phase 1 uses pretrained COCO classes mapped for reports:

- `bicycle`
- `motorcycle`
- `car`
- `bus`
- `truck`

Your requested categories (**bicycle / car / truck**) are covered; `motorcycle` and `bus` are kept as separate detector classes so you do not lose information. India-specific classes (auto rickshaw, MAV, etc.) need Phase 2 fine-tuning — see `config/india_classes.yaml`.

---

## 6. How to configure lines

Edit `config/camera_config.yaml` → `junction:` (or copy from `configs_examples/junction_*.yaml`).

Each arm needs two lines in **pixel coordinates** of the video:

```yaml
junction:
  type: four_way
  arms:
    - name: north
      in_line:  [[x1,y1],[x2,y2]]   # entering from north
      out_line: [[x1,y1],[x2,y2]]   # leaving toward north
    # ... east, south, west
```

**Tips**

1. Open one frame in an image editor; note X/Y for each approach.
2. Draw lines **across** the lane, not along it.
3. Keep IN and OUT lines slightly separated so the same vehicle does not hit both in one hop.
4. After changing YAML, restart the API so new jobs pick up the geometry.

When the React UI selects a junction type:

- If `camera_config.yaml` already has a valid `junction` of that type → those arms are used (your calibrated lines).
- Otherwise the matching preset under `configs_examples/` is applied (placeholder lines — **must calibrate** for real sites).

---

## 7. Using the React dashboard

1. Start API: `python scripts/run_api.py`
2. Start UI: `cd frontend && npm run dev`
3. Open http://127.0.0.1:5173
4. Choose **2-way / 3-way / 4-way**
5. Upload one or many videos → Process
6. Per video: **movement table** (each way) + **class table** + Excel download

API also exposes `GET /api/junction-types` for the UI metadata.

---

## 8. Code map

| File | Responsibility |
|------|----------------|
| `src/junction.py` | Type normalization, 2/6/8 expansion, validation |
| `src/counter.py` | Line crossing + dedupe per movement |
| `src/schemas.py` | `CountEvent` with `arm` / `flow` / `movement_id` |
| `src/exporter.py` | Excel schema for junction reports |
| `api/jobs.py` | Merge junction type into pipeline config per job |
| `api/main.py` | `junction_type` form field on `/api/upload` |
| `frontend/src/App.tsx` | Junction type picker |
| `configs_examples/junction_*.yaml` | Example geometries |

---

## 9. Limits (honest)

- Placeholder line coordinates will **not** match your camera until you calibrate.
- COCO classes ≠ full India ATCC taxonomy.
- This counts **per-arm IN/OUT volumes**, not every left/through/right turn pair (12 movements on a 4-leg signalized junction). Extending to OD turning matrices is a future enhancement using entry-arm → exit-arm track linking.
