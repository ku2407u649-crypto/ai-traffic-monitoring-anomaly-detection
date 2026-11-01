"""Learn what NORMAL traffic looks like on your own camera and save the ML anomaly model.

  python train_anomaly.py --videos normal1.mp4 normal2.mp4 --out models/anomaly_if.joblib

The saved model is loaded automatically by run_monitor.py / the dashboard
(config.yaml -> anomaly.ml.model_path). Use videos that show ordinary traffic only.
"""
import argparse

import numpy as np

from traffic.config import load_config
from traffic.engine import SceneEngine
from traffic.ml import AnomalyModel, track_features
from traffic.monitor import TrafficMonitor
import cv2


def collect(cfg, video, stride):
    mon = TrafficMonitor(cfg)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ow, oh = int(cap.get(3)), int(cap.get(4))
    rw = cfg["video"]["resize_width"]
    up = cfg['video'].get('upscale_small_to') or 0
    W, H = (rw, int(oh * rw / ow)) if (rw and ow > rw) else ((up, int(oh * up / ow)) if (up and ow < up) else (ow, oh))
    cfg2 = {**cfg, "anomaly": {**cfg["anomaly"], "ml": {**cfg["anomaly"]["ml"], "enabled": False}}}
    eng = SceneEngine(cfg2, (W, H))
    feats, idx = [], -1
    ended = {}
    while True:
        if not cap.grab():
            break
        idx += 1
        if idx % stride:
            continue
        ok, fr = cap.retrieve()
        if not ok:
            break
        fr = cv2.resize(fr, (W, H))
        dets = mon.detect_and_track(fr)
        eng.update(idx / fps, idx, dets)
        for tid, tr in list(eng.tracks.items()):
            ended[tid] = tr
    for tr in ended.values():
        if tr.cls != 0 and tr.last_t - tr.first_t >= cfg["anomaly"]["ml"]["min_track_seconds"]:
            f = track_features(tr.hist, eng.diag)
            if f is not None:
                feats.append(f)
    cap.release()
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", default="models/anomaly_if.joblib")
    ap.add_argument("--config", default=None)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    cfg = load_config(a.config)
    X = []
    for v in a.videos:
        f = collect(cfg, v, a.stride)
        print(f"{v}: {len(f)} normal tracks")
        X += f
    if len(X) < 20:
        raise SystemExit("Need at least 20 tracks - use longer / busier videos.")
    ml = AnomalyModel({**cfg["anomaly"]["ml"], "model_path": None})
    ml.fit(np.vstack(X))
    ml.save(a.out)
    print(f"Saved anomaly model trained on {len(X)} tracks -> {a.out}")


if __name__ == "__main__":
    main()
