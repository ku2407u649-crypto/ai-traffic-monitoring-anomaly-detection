# AI-Based Traffic Monitoring and Anomaly Detection System

Working prototype for the B.Tech mini project (UnitedWorld Institute of Technology, Karnavati University).
It watches road-camera video, **detects and tracks vehicles with YOLO + ByteTrack**, counts them, estimates
density / speed, and raises **alerts for anomalies** using rules plus a machine-learning model. Everything is
shown on a Streamlit dashboard and saved as video, CSV, SQLite database and snapshot images.

```
Video / camera  ->  YOLOv8 detector  ->  ByteTrack tracker  ->  Scene engine  ->  Dashboard + logs
                                                               |- vehicle counting (virtual line)
                                                               |- density & congestion
                                                               |- speed (km/h, if calibrated)
                                                               |- rules: stopped vehicle, wrong-way,
                                                               |         over-speeding, pedestrian on road
                                                               '- ML: Isolation Forest on motion features
```

## 1. Install
Python 3.10+ is required. A GPU is optional (the nano model runs at about 10 FPS on one CPU core).

```bash
python -m venv venv
venv\Scripts\activate          # Windows      (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python scripts/download_samples.py      # public sample traffic videos
```
The YOLO weights (`yolov8n.pt`) are downloaded automatically on first run.

## 2. Run
**Dashboard (easiest)**
```bash
streamlit run dashboard.py
```
Upload a video (or type a path / RTSP URL / webcam index), adjust the sliders, press **Analyse video**.
You get live preview, KPIs, charts, alert table with snapshots, the annotated video and CSV downloads.

**Command line**
```bash
python run_monitor.py --source sample_videos/highway_video2.avi
python run_monitor.py --source 0                                  # webcam
python run_monitor.py --source rtsp://user:pass@192.168.1.10/stream --show
python run_monitor.py --source video.mp4 --stride 2               # 2x faster on CPU
```
Outputs go to `outputs/`: `annotated.mp4`, `events.csv`, `events.db` (SQLite), `stats.csv`, `summary.json`, `snapshots/`.

## 3. Configure your camera (`config.yaml`)
All positions are normalised (0-1) so they work at any resolution.

| Setting | What it does |
|---|---|
| `scene.counting_line` | virtual line; every vehicle crossing it is counted once |
| `scene.roi` | polygon of the road area used for density, congestion, stopped and wrong-way rules |
| `scene.expected_direction` | `[0,1]` = traffic flows top->bottom. Set it to enable **wrong-way** detection |
| `scene.calibration` | `scale` (metres per pixel) or `homography` (4 road points) to get **km/h**; needed for over-speeding |
| `scene.pedestrian_zone` | polygon where pedestrians are not allowed (enables the pedestrian rule) |
| `anomaly.*` | thresholds: stop time, speed limit, congestion size, cooldown, ML sensitivity |
| `model.weights` | `yolov8n.pt` (fast) / `yolov8s.pt` (more accurate) / your fine-tuned `best.pt` |

Tip: make one config file per camera (`python run_monitor.py --config junction1.yaml ...`).

## 4. Anomaly detection
| Alert | How it is detected |
|---|---|
| Stopped vehicle | bottom-centre of the vehicle moves less than `stop_move_frac` of the frame width for `stop_seconds`. If 5+ vehicles stop together it is treated as a queue (red light / jam), not an incident |
| Wrong-way | direction of motion differs from `expected_direction` by more than 135 degrees for 1 s, inside the ROI |
| Over-speeding | smoothed speed above `speed_limit_kmh` for 1 s (needs calibration) |
| Sudden congestion | >= `congestion_min_vehicles` in the ROI and mean speed below threshold for several seconds |
| Pedestrian on road | a person inside `pedestrian_zone` for 1 s |
| Unusual motion (ML) | Isolation Forest on per-vehicle features (speed, speed variation, max acceleration, heading change, straightness). It learns "normal" online from the first 40 finished tracks, or from a saved model |

Train the ML model on your own normal traffic (optional but recommended):
```bash
python train_anomaly.py --videos normal1.mp4 normal2.mp4 --out models/anomaly_if.joblib
```

## 5. Improve the detector (fine-tuning)
```bash
python train_yolo.py --data dataset.yaml --epochs 50     # UA-DETRAC / your own labelled data, use a GPU / Colab
```
Then set `model.weights` in `config.yaml` to `runs/detect/traffic/weights/best.pt`.
Evaluate anomaly alerts against ground truth: `python evaluate_events.py --pred outputs/events.csv --truth ground_truth.csv`
(precision, recall and F1 per alert type).

## 6. Tests and measured results
```bash
python -m unittest discover -s tests -v          # 13 tests: counting, every rule, ML, warm-up
```
Measured on a **1-core CPU, no GPU**, `yolov8n`, a public 30 s highway clip (`highway_video2.avi`, 640x480):

* 750 frames processed in 70 s (**about 10.7 FPS**), 18 vehicles counted (13 cars, 5 trucks), ML model warmed up automatically
* `sample_outputs/` contains the summary, per-frame stats and example images
* **Rule check**: `scripts/rule_test.yaml` sets thresholds on purpose so that alerts fire on normal traffic. It produced
  20 over-speeding and 17 wrong-way alerts with snapshots and database rows, confirming the alert pipeline end to end.
  These are *tests*, not real incidents.

No ground-truth labels were available for these clips, so **accuracy numbers (mAP, F1, MOTA) are not claimed here.**
Measure them on a labelled dataset with `train_yolo.py` (mAP) and `evaluate_events.py` (alert F1).

## 7. Known limitations
* `yolov8n` is small; on blurry / far-away vehicles it can miss objects or switch IDs. Use `yolov8s`/`m` or fine-tune.
* Speed in km/h is only as good as the calibration; use the `homography` mode with a known road rectangle.
* Fixed camera assumed. Night, heavy rain and heavy occlusion reduce accuracy.
* Stopped-vehicle rule cannot know *why* a vehicle stopped (breakdown vs. waiting); tune `stop_seconds` per junction.

## 8. Project structure
```
config.yaml            all settings              run_monitor.py     command-line runner
dashboard.py           Streamlit dashboard       train_yolo.py      fine-tune YOLO
train_anomaly.py       learn normal motion       evaluate_events.py precision / recall / F1
traffic/engine.py      counting, speed, rules    traffic/ml.py      Isolation Forest + features
traffic/monitor.py     YOLO + tracker + output   traffic/events.py  SQLite / CSV / snapshots
traffic/draw.py        overlays                  tests/             unit tests
```

## 9. Credits and licences
Ultralytics YOLO is licensed under AGPL-3.0 (check terms for commercial use). Sample videos belong to their
original repositories (see `scripts/download_samples.py`). Team: Parth Solanki, Swayam Tanna, Jay Thakkar.
Mentor: Ms. Pragya Vats.
