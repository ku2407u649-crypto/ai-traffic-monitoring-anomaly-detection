"""Streamlit dashboard for the AI-Based Traffic Monitoring and Anomaly Detection System.

Run:   streamlit run dashboard.py
"""
import json
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from traffic.config import load_config, ROOT
from traffic.monitor import TrafficMonitor

st.set_page_config(page_title="AI Traffic Monitor", page_icon="🚦", layout="wide")

DIRECTIONS = {
    "Off (no wrong-way detection)": None,
    "Top → Bottom": [0, 1],
    "Bottom → Top": [0, -1],
    "Left → Right": [1, 0],
    "Right → Left": [-1, 0],
}
EVENT_LABEL = {
    "stopped_vehicle": "Stopped vehicle", "wrong_way": "Wrong-way driving", "over_speeding": "Over-speeding",
    "congestion": "Sudden congestion", "pedestrian_on_road": "Pedestrian on road", "unusual_motion": "Unusual motion (ML)",
}


@st.cache_resource(show_spinner="Loading YOLO model...")
def get_monitor_model(weights: str):
    from ultralytics import YOLO
    w = weights if Path(weights).exists() else str(ROOT / weights)
    return YOLO(w)


def show_results(out_dir: Path):
    sj, ev_csv, st_csv = out_dir / "summary.json", out_dir / "events.csv", out_dir / "stats.csv"
    if not sj.exists():
        st.info("No results found in this folder yet.")
        return
    summary = json.loads(sj.read_text(encoding="utf-8"))
    stats = pd.read_csv(st_csv) if st_csv.exists() else pd.DataFrame()
    events = pd.read_csv(ev_csv) if ev_csv.exists() else pd.DataFrame()

    c = st.columns(5)
    c[0].metric("Vehicles counted", summary["vehicle_count"]["total"])
    c[1].metric("Unique vehicles tracked", summary["unique_tracks"])
    c[2].metric("Alerts raised", summary["events_total"])
    c[3].metric("Peak vehicles in ROI", int(stats["vehicles_in_roi"].max()) if len(stats) else 0)
    c[4].metric("Processing speed", f"{summary['avg_fps']} FPS")

    t1, t2, t3, t4 = st.tabs(["📈 Traffic analytics", "🚨 Alerts", "🎬 Annotated video", "⬇️ Downloads"])
    with t1:
        a, b = st.columns([2, 1])
        with a:
            st.subheader("Traffic density over time")
            if len(stats):
                st.line_chart(stats.set_index("t")[["vehicles_in_roi", "vehicles_in_frame"]])
            st.subheader("Average speed (normalised)")
            if len(stats):
                st.line_chart(stats.set_index("t")[["mean_speed_norm"]])
        with b:
            st.subheader("Vehicles by class")
            by = summary["vehicle_count"]["by_class"]
            if by:
                st.bar_chart(pd.Series(by, name="count"))
            else:
                st.caption("No vehicle crossed the counting line.")
            st.subheader("Direction")
            st.write(f"Forward: **{summary['vehicle_count']['forward']}**  |  "
                     f"Reverse: **{summary['vehicle_count']['reverse']}**")
            st.subheader("Alerts by type")
            if summary["events_by_type"]:
                st.bar_chart(pd.Series({EVENT_LABEL.get(k, k): v for k, v in summary["events_by_type"].items()}))
            else:
                st.success("No anomalies detected.")
    with t2:
        if len(events):
            show = events.copy()
            show["type"] = show["type"].map(lambda x: EVENT_LABEL.get(x, x))
            st.dataframe(show[["t", "type", "track_id", "cls", "detail", "severity"]])
            st.subheader("Snapshots")
            cols = st.columns(3)
            for i, row in events.reset_index(drop=True).iterrows():
                p = out_dir / str(row["snapshot"])
                if row["snapshot"] and p.exists():
                    cols[i % 3].image(str(p), caption=f"{EVENT_LABEL.get(row['type'], row['type'])} · t={row['t']}s · #{row['track_id']}")
        else:
            st.success("No anomalies detected in this video.")
    with t3:
        v = out_dir / "annotated.mp4"
        if v.exists():
            st.video(str(v))
        else:
            st.info("Annotated video was not saved.")
    with t4:
        for name, mime in [("events.csv", "text/csv"), ("stats.csv", "text/csv"), ("summary.json", "application/json"),
                           ("annotated.mp4", "video/mp4")]:
            p = out_dir / name
            if p.exists():
                st.download_button(f"Download {name}", p.read_bytes(), file_name=name, mime=mime)


# ----------------------------------------------------------------------------- UI
st.title("🚦 AI-Based Traffic Monitoring and Anomaly Detection")
st.caption("YOLO vehicle detection · ByteTrack tracking · rule-based + ML anomaly detection")

base = load_config()
with st.sidebar:
    st.header("Video source")
    mode = st.radio("Choose", ["Upload a video", "Path / URL / webcam"], label_visibility="collapsed")
    upload, path_txt = None, ""
    if mode == "Upload a video":
        upload = st.file_uploader("Traffic video", type=["mp4", "avi", "mov", "mkv"])
    else:
        path_txt = st.text_input("File path, RTSP URL or webcam index", "sample_videos/car-detection.mp4")

    st.header("Detection")
    conf = st.slider("Confidence threshold", 0.1, 0.8, float(base["model"]["conf"]), 0.05)
    stride = st.select_slider("Speed vs accuracy (analyse every N-th frame)", [1, 2, 3, 4], value=2)

    st.header("Scene")
    line_y = st.slider("Counting line position (top → bottom)", 0.2, 0.9, 0.62, 0.02)
    direction = st.selectbox("Normal traffic direction", list(DIRECTIONS), index=0)
    mpp = st.number_input("Metres per pixel (0 = speed in km/h off)", 0.0, 1.0, 0.0, 0.005, format="%.3f")

    st.header("Anomaly rules")
    stop_s = st.slider("Stopped vehicle after (seconds)", 2, 20, int(base["anomaly"]["stop_seconds"]))
    limit = st.slider("Speed limit (km/h)", 20, 120, int(base["anomaly"]["speed_limit_kmh"]), 5)
    ml_on = st.checkbox("ML unusual-motion detection", value=True)
    run_btn = st.button("▶ Analyse video", type="primary")

if run_btn:
    if mode == "Upload a video":
        if upload is None:
            st.error("Please upload a video first.")
            st.stop()
        tmp = Path(tempfile.mkdtemp()) / upload.name
        tmp.write_bytes(upload.getbuffer())
        source = str(tmp)
    else:
        source = path_txt.strip()
        if source and not source.isdigit() and not source.startswith(("rtsp://", "http")) and not Path(source).exists():
            if (ROOT / source).exists():
                source = str(ROOT / source)
            else:
                st.error(f"File not found: {source}")
                st.stop()

    ov = {
        "model": {"conf": conf},
        "video": {"process_every_n": stride},
        "scene": {"counting_line": [[0.05, line_y], [0.95, line_y]], "expected_direction": DIRECTIONS[direction],
                  "calibration": {"mode": "scale" if mpp > 0 else "none", "meters_per_pixel": mpp or 0.05}},
        "anomaly": {"stop_seconds": float(stop_s), "speed_limit_kmh": float(limit), "ml": {"enabled": ml_on}},
    }
    cfg = load_config(None, ov)
    out_dir = ROOT / "outputs" / "dashboard"
    mon = TrafficMonitor(cfg, out_dir)
    mon._model = get_monitor_model(cfg["model"]["weights"])

    st.subheader("Live analysis")
    bar = st.progress(0.0, text="Starting...")
    left, right = st.columns([3, 1])
    frame_slot = left.empty()
    kpi = right.empty()
    t0, n = time.time(), [0]

    def on_frame(p, frame, stats):
        n[0] += 1
        if n[0] % 2 == 0:
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            kpi.markdown(f"**Vehicles in ROI:** {stats['vehicles_in_roi']}  \n**Counted:** {stats['total_count']}  \n"
                         f"**Density:** {stats['density']}  \n**Pedestrians:** {stats['pedestrians']}")
        bar.progress(min(max(p, 0.0), 1.0), text=f"Analysing... {p * 100:.0f}%")

    with st.spinner("Running detection and tracking..."):
        summary = mon.run(source, progress_cb=on_frame)
    bar.progress(1.0, text="Done")
    frame_slot.empty(); kpi.empty()
    st.session_state["out_dir"] = str(out_dir)
    st.success(f"Finished: {summary['frames_processed']} frames in {summary['processing_seconds']} s "
               f"({summary['avg_fps']} FPS) · {summary['events_total']} alert(s)")

st.divider()
prev = st.session_state.get("out_dir") or str(ROOT / "outputs" / "dashboard")
folder = st.text_input("Results folder", prev, help="Open any folder created by run_monitor.py")
if Path(folder).exists():
    show_results(Path(folder))
