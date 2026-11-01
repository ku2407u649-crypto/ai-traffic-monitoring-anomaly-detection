"""TrafficMonitor: YOLO detection + ByteTrack tracking + SceneEngine + outputs."""
from __future__ import annotations
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .config import ROOT
from .draw import draw_scene
from .engine import SceneEngine
from .events import EventStore
from .ml import AnomalyModel


def resolve_device(dev) -> str | None:
    if dev in (None, "", "auto"):
        return None
    return str(dev)


class TrafficMonitor:
    def __init__(self, cfg: dict, out_dir: str | Path | None = None):
        self.cfg = cfg
        self.out_dir = Path(out_dir or cfg["output"]["dir"])
        self._model = None

    # ------------------------------------------------------------ model
    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO  # imported lazily so the engine can be tested without PyTorch
            w = self.cfg["model"]["weights"]
            if not Path(w).exists() and (ROOT / w).exists():
                w = str(ROOT / w)
            self._model = YOLO(w)
        return self._model

    def detect_and_track(self, frame):
        m, mc = self._load_model(), self.cfg["model"]
        classes = list(mc["vehicle_classes"]) + ([0] if mc.get("detect_person", True) else [])
        res = m.track(frame, persist=True, tracker=mc["tracker"], conf=mc["conf"], iou=mc["iou"],
                      imgsz=mc["imgsz"], classes=classes, device=resolve_device(mc.get("device")), verbose=False)[0]
        dets = []
        if res.boxes is not None and res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().numpy()
            cls = res.boxes.cls.int().cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            for b, i, c, s in zip(xyxy, ids, cls, conf):
                dets.append({"id": int(i), "cls": int(c), "box": tuple(float(v) for v in b), "conf": float(s)})
        return dets

    # ------------------------------------------------------------ main loop
    def run(self, source, max_frames: int | None = None,
            progress_cb: Callable[[float, np.ndarray, dict], None] | None = None,
            show: bool = False) -> dict:
        cfg, vcfg, ocfg = self.cfg, self.cfg["video"], self.cfg["output"]
        src = int(source) if str(source).isdigit() else str(source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {source}")
        is_file = isinstance(src, str) and not src.startswith(("rtsp://", "http://", "https://"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if is_file else 0
        fps_eff = fps if fps > 1 else 25.0
        ow, oh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rw = int(vcfg.get("resize_width") or 0)
        up = int(vcfg.get("upscale_small_to") or 0)
        if rw and ow > rw:                    # large video  -> down-scale for speed
            W, H = rw, int(round(oh * rw / ow))
        elif up and 0 < ow < up:              # tiny video   -> up-scale so small vehicles / text stay readable
            W, H = up, int(round(oh * up / ow))
        else:
            W, H = ow, oh
        stride = max(1, int(vcfg.get("process_every_n", 1)))

        self.out_dir.mkdir(parents=True, exist_ok=True)
        store = EventStore(self.out_dir, ocfg.get("save_snapshots", True))
        ml = AnomalyModel(cfg["anomaly"].get("ml", {}))
        engine = SceneEngine(cfg, (W, H), ml)
        writer = None
        video_path = self.out_dir / "annotated_raw.mp4"
        if ocfg.get("save_video", True):
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_eff / stride, (W, H))

        stats_rows, banner, banner_until = [], None, 0.0
        t_wall0, idx, done = time.time(), -1, 0
        last_stats = None
        try:
            while True:
                ok = cap.grab()
                if not ok:
                    break
                idx += 1
                if idx % stride:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if (W, H) != (ow, oh):
                    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                t = idx / fps_eff if is_file else time.time() - t_wall0
                dets = self.detect_and_track(frame)
                stats, new_events = engine.update(t, idx, dets)
                if new_events:
                    banner = " | ".join(f"{e.type.replace('_', ' ').upper()} (#{e.track_id})" if e.track_id >= 0
                                        else e.type.replace('_', ' ').upper() for e in new_events)
                    banner_until = t + 3.0
                fps_text = f"{done / max(time.time() - t_wall0, 1e-6):4.1f} FPS"
                out = draw_scene(frame, engine, dets, stats, t, fps_text, banner if t <= banner_until else None)
                for e in new_events:
                    store.add(e, out)
                if writer is not None:
                    writer.write(out)
                stats_rows.append(stats)
                last_stats = stats
                done += 1
                if progress_cb:
                    progress_cb(min(1.0, idx / n_total) if n_total else 0.0, out, stats)
                if show:
                    cv2.imshow("AI Traffic Monitor (q to quit)", out)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if max_frames and done >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        engine.finalize()
        elapsed = time.time() - t_wall0
        # --- outputs
        pd.DataFrame(stats_rows).to_csv(self.out_dir / "stats.csv", index=False)
        store.export_csv()
        store.close()
        if writer is not None:
            final = self.out_dir / "annotated.mp4"
            self._to_h264(video_path, final)
        by_type = pd.Series([e.type for e in store.rows]).value_counts().to_dict() if store.rows else {}
        summary = {
            "source": str(source), "frames_processed": done, "processing_seconds": round(elapsed, 1),
            "avg_fps": round(done / max(elapsed, 1e-6), 2), "resolution": [W, H],
            "vehicle_count": engine.counts, "unique_tracks": len(engine.ids_seen),
            "events_total": len(store.rows), "events_by_type": by_type,
            "ml_model_fitted": engine.ml.is_fitted,
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.engine = engine
        return summary

    @staticmethod
    def _to_h264(src: Path, dst: Path) -> None:
        """Re-encode to H.264 so the video plays in browsers / Streamlit (needs ffmpeg)."""
        ff = shutil.which("ffmpeg")
        if ff:
            r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-vcodec", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "26", str(dst)])
            if r.returncode == 0:
                src.unlink(missing_ok=True)
                return
        shutil.move(str(src), str(dst))
