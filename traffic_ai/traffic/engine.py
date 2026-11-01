"""Scene engine: tracking state, vehicle counting, speed estimation and anomaly rules.

The engine is independent of the detector. It receives, for every processed frame,
a list of tracked detections and returns statistics plus any new anomaly events.
That makes it easy to unit-test with synthetic tracks (see tests/test_engine.py).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import cv2

from .ml import AnomalyModel, track_features

CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

EVENT_STOPPED = "stopped_vehicle"
EVENT_WRONG_WAY = "wrong_way"
EVENT_OVERSPEED = "over_speeding"
EVENT_CONGESTION = "congestion"
EVENT_PEDESTRIAN = "pedestrian_on_road"
EVENT_UNUSUAL = "unusual_motion"


@dataclass
class Event:
    t: float
    frame: int
    type: str
    track_id: int
    cls: str
    detail: str
    severity: str = "warning"
    snapshot: str = ""


class Track:
    def __init__(self, tid: int, cls: int, t: float, frame: int):
        self.id, self.cls, self.first_t, self.last_t, self.first_frame = tid, cls, t, t, frame
        self.pts: list[tuple[float, float, float]] = []      # recent (t, x, y) used by the rules
        self.hist: list[tuple[float, float, float]] = []     # thinned full history used by the ML model
        self.box = (0.0, 0.0, 0.0, 0.0)
        self.prev: tuple[float, float] | None = None
        self.speed_norm = 0.0
        self.speed_kmh: float | None = None
        self.counted = False
        self.stopped = False
        self.wrong_since: float | None = None
        self.over_since: float | None = None
        self.ped_since: float | None = None
        self.ml_hits = 0
        self.last_ml = -1e9
        self.alerted: dict[str, float] = {}

    @property
    def pos(self) -> tuple[float, float]:
        return self.pts[-1][1], self.pts[-1][2]

    def age(self, t: float) -> float:
        return t - self.first_t


def _ccw(a, b, c) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def seg_intersect(p1, p2, a, b) -> bool:
    return _ccw(p1, a, b) != _ccw(p2, a, b) and _ccw(p1, p2, a) != _ccw(p1, p2, b)


def side(a, b, p) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


class SceneEngine:
    TRACK_TIMEOUT = 3.0      # seconds without detection before a track is closed

    def __init__(self, cfg: dict, frame_size: tuple[int, int], ml: AnomalyModel | None = None):
        self.w, self.h = frame_size
        self.diag = math.hypot(self.w, self.h)
        sc, an = cfg["scene"], cfg["anomaly"]
        self.an = an
        px = lambda poly: np.array(poly, dtype=float) * [self.w, self.h]
        self.line = px(sc["counting_line"]) if sc.get("counting_line") else None
        self.roi = px(sc["roi"]).astype(np.float32) if sc.get("roi") else None
        self.ped_zone = px(sc["pedestrian_zone"]).astype(np.float32) if sc.get("pedestrian_zone") else None
        ed = sc.get("expected_direction")
        self.expected = None
        if ed:
            v = np.array(ed, dtype=float)
            self.expected = v / (np.linalg.norm(v) + 1e-9)
        # speed calibration
        cal = sc.get("calibration", {}) or {}
        self.cal_mode = cal.get("mode", "none")
        self.mpp = float(cal.get("meters_per_pixel", 0.05))
        self.H = None
        if self.cal_mode == "homography" and len(cal.get("homography_src", [])) == 4:
            src = (np.array(cal["homography_src"], dtype=np.float32) * [self.w, self.h]).astype(np.float32)
            dst = np.array(cal["homography_dst"], dtype=np.float32)
            self.H = cv2.getPerspectiveTransform(src, dst)
        self.stop_move_px = an["stop_move_frac"] * self.w
        self.hist_seconds = max(an["stop_seconds"], 3.0) + 2.0
        self.ml = ml or AnomalyModel(an.get("ml", {}))
        self.ml_cfg = an.get("ml", {})

        self.tracks: dict[int, Track] = {}
        self.events: list[Event] = []
        self.flagged: dict[int, tuple[float, str]] = {}
        self.counts = {"total": 0, "forward": 0, "reverse": 0, "by_class": {}}
        self.cong_since: float | None = None
        self.last_congestion_alert = -1e9
        self.ids_seen: set[int] = set()

    # ------------------------------------------------------------------ helpers
    def _in_poly(self, poly, p) -> bool:
        return poly is None or cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0

    def _world(self, p):
        if self.cal_mode == "scale":
            return np.array([p[0] * self.mpp, p[1] * self.mpp])
        if self.H is not None:
            q = cv2.perspectiveTransform(np.array([[p]], dtype=np.float32), self.H)[0, 0]
            return q.astype(float)
        return None

    def _can_alert(self, tr: Track, rule: str, t: float) -> bool:
        last = tr.alerted.get(rule)
        return last is None or t - last >= self.an["cooldown_seconds"]

    def _emit(self, t, frame, typ, tr: Track | None, detail, severity="warning") -> Event:
        ev = Event(t=round(t, 2), frame=frame, type=typ, track_id=tr.id if tr else -1,
                   cls=CLASS_NAMES.get(tr.cls, f"class{tr.cls}") if tr else "-", detail=detail, severity=severity)
        self.events.append(ev)
        if tr:
            tr.alerted[typ] = t
            self.flagged[tr.id] = (t + 3.0, typ)
        return ev

    # ------------------------------------------------------------------ per-track updates
    def _update_speed(self, tr: Track, t: float) -> None:
        i = len(tr.pts) - 1
        while i > 0 and t - tr.pts[i - 1][0] <= 1.0:
            i -= 1
        t0, x0, y0 = tr.pts[i]
        t1, x1, y1 = tr.pts[-1]
        dt = t1 - t0
        if dt < 0.4:
            return
        sn = math.hypot(x1 - x0, y1 - y0) / dt / self.diag
        tr.speed_norm = sn if tr.speed_norm == 0 else 0.5 * tr.speed_norm + 0.5 * sn
        w0, w1 = self._world((x0, y0)), self._world((x1, y1))
        if w0 is not None and w1 is not None:
            kmh = float(np.linalg.norm(w1 - w0)) / dt * 3.6
            tr.speed_kmh = kmh if tr.speed_kmh is None else 0.5 * tr.speed_kmh + 0.5 * kmh

    def _check_count(self, tr: Track, cur) -> None:
        if self.line is None or tr.counted or tr.prev is None:
            return
        a, b = self.line
        if seg_intersect(tr.prev, cur, a, b):
            direction = "forward" if side(a, b, tr.prev) < 0 <= side(a, b, cur) else "reverse"
            tr.counted = True
            self.counts["total"] += 1
            self.counts[direction] += 1
            name = CLASS_NAMES.get(tr.cls, f"class{tr.cls}")
            self.counts["by_class"][name] = self.counts["by_class"].get(name, 0) + 1

    def _is_stopped(self, tr: Track, t: float) -> bool:
        need = self.an["stop_seconds"]
        if tr.age(t) < need:
            return False
        win = [p for p in tr.pts if p[0] >= t - need]
        if len(win) < 5 or win[0][0] > t - need + 1.0:
            return False
        xs, ys = [p[1] for p in win], [p[2] for p in win]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) < self.stop_move_px

    def _is_wrong_way(self, tr: Track, t: float) -> tuple[bool, float]:
        if self.expected is None:
            return False, 0.0
        win = [p for p in tr.pts if p[0] >= t - 2.0]
        if len(win) < 5 or win[-1][0] - win[0][0] < 0.8:
            return False, 0.0
        m = np.array([win[-1][1] - win[0][1], win[-1][2] - win[0][2]])
        n = float(np.linalg.norm(m))
        if n < self.an["wrong_way_min_move_frac"] * self.w:
            return False, 0.0
        ang = math.degrees(math.acos(float(np.clip(m @ self.expected / n, -1, 1))))
        return ang >= self.an["wrong_way_angle_deg"], ang

    # ------------------------------------------------------------------ main update
    def update(self, t: float, frame_idx: int, dets: list[dict]) -> tuple[dict, list[Event]]:
        new_events: list[Event] = []
        seen = set()
        for d in dets:
            tid, cls = int(d["id"]), int(d["cls"])
            x1, y1, x2, y2 = d["box"]
            cur = ((x1 + x2) / 2.0, float(y2))             # bottom-centre of the box touches the road
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = Track(tid, cls, t, frame_idx)
                self.ids_seen.add(tid)
            tr.box, tr.last_t = (x1, y1, x2, y2), t
            tr.pts.append((t, cur[0], cur[1]))
            while tr.pts and t - tr.pts[0][0] > self.hist_seconds:
                tr.pts.pop(0)
            if not tr.hist or t - tr.hist[-1][0] >= 0.2:
                tr.hist.append((t, cur[0], cur[1]))
                if len(tr.hist) > 600:
                    tr.hist.pop(0)
            seen.add(tid)
            if cls != 0:
                self._update_speed(tr, t)
                self._check_count(tr, cur)
            tr.prev = cur

        # close tracks that disappeared
        for tid in [i for i, tr in self.tracks.items() if t - tr.last_t > self.TRACK_TIMEOUT]:
            self._close_track(self.tracks.pop(tid))

        active = [self.tracks[i] for i in seen]
        vehicles = [tr for tr in active if tr.cls != 0]
        in_roi = [tr for tr in vehicles if self._in_poly(self.roi, tr.pos)]

        # ---- rule: stopped vehicle (with queue suppression)
        for tr in vehicles:
            tr.stopped = self._is_stopped(tr, t)
        stopped_roi = [tr for tr in in_roi if tr.stopped]
        queue = len(stopped_roi) >= self.an["queue_suppress_count"]
        for tr in stopped_roi:
            if not queue and self._can_alert(tr, EVENT_STOPPED, t):
                new_events.append(self._emit(t, frame_idx, EVENT_STOPPED, tr,
                                             f"Stationary for more than {self.an['stop_seconds']:.0f}s", "critical"))

        for tr in vehicles:
            # ---- rule: wrong-way driving
            wrong, ang = self._is_wrong_way(tr, t) if tr in in_roi else (False, 0.0)   # only inside the monitored road area
            if wrong:
                tr.wrong_since = tr.wrong_since or t
                if t - tr.wrong_since >= self.an["wrong_way_seconds"] and self._can_alert(tr, EVENT_WRONG_WAY, t):
                    new_events.append(self._emit(t, frame_idx, EVENT_WRONG_WAY, tr,
                                                 f"Moving {ang:.0f} deg against the expected direction", "critical"))
            else:
                tr.wrong_since = None
            # ---- rule: over-speeding (only if calibrated)
            if tr.speed_kmh is not None and tr.speed_kmh > self.an["speed_limit_kmh"]:
                tr.over_since = tr.over_since or t
                if t - tr.over_since >= self.an["overspeed_seconds"] and self._can_alert(tr, EVENT_OVERSPEED, t):
                    new_events.append(self._emit(t, frame_idx, EVENT_OVERSPEED, tr,
                                                 f"{tr.speed_kmh:.0f} km/h (limit {self.an['speed_limit_kmh']})"))
            else:
                tr.over_since = None
            # ---- ML: unusual motion
            ev = self._ml_check(tr, t, frame_idx)
            if ev:
                new_events.append(ev)

        # ---- rule: pedestrian on carriageway
        if self.ped_zone is not None:
            for tr in active:
                if tr.cls == 0 and self._in_poly(self.ped_zone, tr.pos):
                    tr.ped_since = tr.ped_since or t
                    if t - tr.ped_since >= self.an["pedestrian_seconds"] and self._can_alert(tr, EVENT_PEDESTRIAN, t):
                        new_events.append(self._emit(t, frame_idx, EVENT_PEDESTRIAN, tr,
                                                     "Pedestrian inside the vehicle-only zone", "critical"))
                else:
                    tr.ped_since = None

        # ---- rule: sudden congestion (global)
        mature = [tr for tr in in_roi if tr.age(t) >= 1.0]
        mean_sn = float(np.mean([tr.speed_norm for tr in mature])) if mature else 0.0
        congested = len(in_roi) >= self.an["congestion_min_vehicles"] and mean_sn < self.an["congestion_max_speed_norm"]
        if congested:
            self.cong_since = self.cong_since or t
            if (t - self.cong_since >= self.an["congestion_seconds"]
                    and t - self.last_congestion_alert >= self.an["cooldown_seconds"]):
                self.last_congestion_alert = t
                new_events.append(self._emit(t, frame_idx, EVENT_CONGESTION, None,
                                             f"{len(in_roi)} vehicles in ROI, mean speed {mean_sn:.3f} (norm.)"))
        else:
            self.cong_since = None

        kmh = [tr.speed_kmh for tr in in_roi if tr.speed_kmh is not None]
        n = len(in_roi)
        mv = self.an["congestion_min_vehicles"]
        stats = {
            "t": round(t, 2), "vehicles_in_frame": len(vehicles), "vehicles_in_roi": n,
            "mean_speed_norm": round(mean_sn, 4),
            "mean_speed_kmh": round(float(np.mean(kmh)), 1) if kmh else None,
            "total_count": self.counts["total"],
            "density": "HIGH" if n >= mv else ("MEDIUM" if n >= mv * 0.5 else "LOW"),
            "pedestrians": sum(1 for tr in active if tr.cls == 0),
        }
        return stats, new_events

    # ------------------------------------------------------------------ ML helpers
    def _ml_check(self, tr: Track, t: float, frame_idx: int) -> Event | None:
        c = self.ml_cfg
        if not self.ml.enabled or not self.ml.is_fitted:
            return None
        if tr.age(t) < c.get("min_track_seconds", 1.5) or t - tr.last_ml < c.get("check_interval_seconds", 1.0):
            return None
        tr.last_ml = t
        feat = track_features(tr.hist, self.diag)
        if feat is None:
            return None
        bad, score = self.ml.score(feat)
        tr.ml_hits = tr.ml_hits + 1 if bad else 0
        if tr.ml_hits >= c.get("consecutive_hits", 2) and self._can_alert(tr, EVENT_UNUSUAL, t):
            return self._emit(t, frame_idx, EVENT_UNUSUAL, tr, f"Motion pattern unlike normal traffic (score {score:.3f})")
        return None

    def _close_track(self, tr: Track) -> None:
        """Finished tracks are used to learn the 'normal' behaviour during warm-up."""
        if tr.cls == 0 or self.ml.is_fitted or not self.ml.enabled:
            return
        if tr.last_t - tr.first_t < self.ml_cfg.get("min_track_seconds", 1.5):
            return
        feat = track_features(tr.hist, self.diag)
        if feat is not None:
            self.ml.add_sample(feat)

    def finalize(self) -> None:
        for tr in list(self.tracks.values()):
            self._close_track(tr)
        self.tracks.clear()

    def is_flagged(self, tid: int, t: float) -> str | None:
        f = self.flagged.get(tid)
        return f[1] if f and f[0] >= t else None
