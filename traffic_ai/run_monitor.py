"""Command-line runner.

Examples
--------
  python run_monitor.py --source sample_videos/traffic.mp4
  python run_monitor.py --source 0                      # webcam
  python run_monitor.py --source rtsp://user:pass@ip/stream --show
  python run_monitor.py --source video.mp4 --config my_junction.yaml --out outputs/junction1
"""
import argparse
import json
import sys

from traffic.config import load_config
from traffic.monitor import TrafficMonitor


def main():
    ap = argparse.ArgumentParser(description="AI-based traffic monitoring and anomaly detection")
    ap.add_argument("--source", required=True, help="video file, webcam index (0) or RTSP/HTTP URL")
    ap.add_argument("--config", default=None, help="YAML config (defaults to config.yaml)")
    ap.add_argument("--out", default=None, help="output folder (default: outputs/)")
    ap.add_argument("--weights", default=None, help="YOLO weights (default from config)")
    ap.add_argument("--conf", type=float, default=None, help="detection confidence")
    ap.add_argument("--stride", type=int, default=None, help="process every N-th frame (faster on CPU)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="do not write the annotated video")
    ap.add_argument("--show", action="store_true", help="show a live preview window (press q to quit)")
    a = ap.parse_args()

    ov = {"model": {}, "video": {}, "output": {}}
    if a.weights: ov["model"]["weights"] = a.weights
    if a.conf is not None: ov["model"]["conf"] = a.conf
    if a.stride: ov["video"]["process_every_n"] = a.stride
    if a.no_video: ov["output"]["save_video"] = False
    cfg = load_config(a.config, ov)

    def progress(p, frame, stats):
        if p:
            print(f"\r  processing... {p * 100:5.1f}%   vehicles in ROI: {stats['vehicles_in_roi']:3d}   "
                  f"count: {stats['total_count']:4d}", end="", flush=True)

    mon = TrafficMonitor(cfg, a.out)
    summary = mon.run(a.source, max_frames=a.max_frames, progress_cb=progress, show=a.show)
    print("\n\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nOutputs saved in: {mon.out_dir.resolve()}")
    print("  annotated.mp4, events.csv, events.db, stats.csv, summary.json, snapshots/")


if __name__ == "__main__":
    sys.exit(main())
