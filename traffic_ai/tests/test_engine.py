"""Unit tests for the anomaly rules, counting and ML module (no GPU / PyTorch needed).

Run:  python -m unittest discover -s tests -v
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traffic.config import load_config
from traffic.engine import SceneEngine, EVENT_STOPPED, EVENT_WRONG_WAY, EVENT_OVERSPEED, EVENT_PEDESTRIAN, \
    EVENT_CONGESTION, EVENT_UNUSUAL
from traffic.ml import AnomalyModel, track_features

W, H, FPS = 960, 540, 10


def det(tid, x, y, cls=2):
    """Detection whose bottom-centre is (x, y)."""
    return {"id": tid, "cls": cls, "box": (x - 30, y - 40, x + 30, y), "conf": 0.9}


def make_engine(**over):
    cfg = load_config()
    cfg["anomaly"]["ml"]["enabled"] = False
    for k, v in over.items():
        sect, key = k.split("__")
        cfg[sect][key] = v
    return SceneEngine(cfg, (W, H))


def run(engine, frames, seconds):
    """frames(t) -> list of detections. Returns all events."""
    ev = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        _, e = engine.update(t, i, frames(t))
        ev += e
    return ev


class TestRules(unittest.TestCase):
    def test_counting_forward_once(self):
        eng = make_engine()
        # vehicle moves downwards across the counting line at y = 0.62 * 540 = 335
        run(eng, lambda t: [det(1, 300, 200 + 100 * t)], 4)
        self.assertEqual(eng.counts["total"], 1)
        self.assertEqual(eng.counts["by_class"], {"car": 1})
        self.assertEqual(eng.counts["forward"] + eng.counts["reverse"], 1)

    def test_counting_both_directions(self):
        eng = make_engine()
        run(eng, lambda t: [det(1, 300, 200 + 100 * t), det(2, 600, 500 - 100 * t)], 4)
        self.assertEqual(eng.counts["total"], 2)
        self.assertEqual(eng.counts["forward"], 1)
        self.assertEqual(eng.counts["reverse"], 1)

    def test_stopped_vehicle_single_alert(self):
        eng = make_engine()
        ev = run(eng, lambda t: [det(1, 480, 400)], 12)
        stopped = [e for e in ev if e.type == EVENT_STOPPED]
        self.assertEqual(len(stopped), 1)
        self.assertGreaterEqual(stopped[0].t, 5.0)

    def test_moving_vehicle_not_stopped(self):
        eng = make_engine()
        ev = run(eng, lambda t: [det(1, 100 + 60 * t, 400)], 12)
        self.assertFalse([e for e in ev if e.type == EVENT_STOPPED])

    def test_queue_is_not_reported_as_incident(self):
        eng = make_engine()
        ev = run(eng, lambda t: [det(i, 100 + 120 * i, 400) for i in range(6)], 12)
        self.assertFalse([e for e in ev if e.type == EVENT_STOPPED])

    def test_wrong_way(self):
        eng = make_engine(scene__expected_direction=[0, 1])           # normal flow: top -> bottom
        ev = run(eng, lambda t: [det(1, 300, 520 - 60 * t), det(2, 600, 100 + 60 * t)], 6)
        ww = [e for e in ev if e.type == EVENT_WRONG_WAY]
        self.assertEqual([e.track_id for e in ww], [1])

    def test_overspeed_needs_calibration_and_speed(self):
        cal = {"mode": "scale", "meters_per_pixel": 0.05, "homography_src": [], "homography_dst": []}
        eng = make_engine(scene__calibration=cal)
        # fast car: 400 px/s * 0.05 m/px = 20 m/s = 72 km/h ; slow car: 100 px/s = 18 km/h
        ev = run(eng, lambda t: [det(1, 40 + 400 * t, 300), det(2, 40 + 100 * t, 450)], 2)
        sp = [e for e in ev if e.type == EVENT_OVERSPEED]
        self.assertEqual([e.track_id for e in sp], [1])
        # without calibration the rule must stay silent
        eng2 = make_engine()
        ev2 = run(eng2, lambda t: [det(1, 40 + 400 * t, 300)], 2)
        self.assertFalse([e for e in ev2 if e.type == EVENT_OVERSPEED])

    def test_pedestrian_in_forbidden_zone(self):
        eng = make_engine(scene__pedestrian_zone=[[0.3, 0.5], [0.7, 0.5], [0.7, 1.0], [0.3, 1.0]])
        ev = run(eng, lambda t: [det(9, 480, 450, cls=0), det(10, 50, 450, cls=0)], 4)
        ped = [e for e in ev if e.type == EVENT_PEDESTRIAN]
        self.assertEqual([e.track_id for e in ped], [9])

    def test_congestion(self):
        eng = make_engine()
        pos = [(80 + 75 * (i % 12), 350 + 40 * (i // 12)) for i in range(12)]
        ev = run(eng, lambda t: [det(i, x + 2 * np.sin(t), y) for i, (x, y) in enumerate(pos)], 10)
        self.assertTrue([e for e in ev if e.type == EVENT_CONGESTION])

    def test_free_flow_no_congestion(self):
        eng = make_engine()
        ev = run(eng, lambda t: [det(i, 50 + 120 * t + 60 * i, 350 + 20 * i) for i in range(12)], 6)
        self.assertFalse([e for e in ev if e.type == EVENT_CONGESTION])


class TestML(unittest.TestCase):
    @staticmethod
    def normal_track(rng):
        v = rng.uniform(90, 130)
        x0, y0 = rng.uniform(50, 400), rng.uniform(150, 450)
        return [(i / FPS, x0 + v * i / FPS, y0 + 0.1 * v * i / FPS + rng.normal(0, 0.3)) for i in range(50)]

    @staticmethod
    def erratic_track():
        pts, x, y = [], 300.0, 300.0
        for i in range(60):
            t = i / FPS
            speed = 40 if (i // 5) % 2 == 0 else 330          # sudden stop-and-go
            ang = 1.2 * np.sin(i * 0.9)                         # swerving
            x += speed / FPS * np.cos(ang)
            y += speed / FPS * np.sin(ang)
            pts.append((t, min(max(x, 20), 940), min(max(y, 20), 520)))
        return pts

    def test_isolation_forest_flags_erratic_motion(self):
        rng = np.random.default_rng(1)
        diag = float(np.hypot(W, H))
        X = np.vstack([track_features(self.normal_track(rng), diag) for _ in range(80)])
        cfg = load_config()["anomaly"]["ml"]
        ml = AnomalyModel({**cfg, "model_path": None, "contamination": 0.02})
        ml.fit(X)
        ok = sum(not ml.score(track_features(self.normal_track(rng), diag))[0] for _ in range(50))
        self.assertGreaterEqual(ok, 45)                        # >= 90 % of normal tracks pass
        bad, _ = ml.score(track_features(self.erratic_track(), diag))
        self.assertTrue(bad)

    def test_engine_emits_unusual_motion(self):
        rng = np.random.default_rng(2)
        diag = float(np.hypot(W, H))
        X = np.vstack([track_features(self.normal_track(rng), diag) for _ in range(80)])
        cfg = load_config()
        cfg["anomaly"]["ml"].update({"enabled": True, "model_path": None, "contamination": 0.02})
        ml = AnomalyModel(cfg["anomaly"]["ml"])
        ml.fit(X)
        eng = SceneEngine(cfg, (W, H), ml)
        pts = self.erratic_track()
        ev = []
        for i, (t, x, y) in enumerate(pts):
            _, e = eng.update(t, i, [det(1, x, y)])
            ev += e
        self.assertTrue([e for e in ev if e.type == EVENT_UNUSUAL])

    def test_online_warmup_fits_model(self):
        rng = np.random.default_rng(3)
        cfg = load_config()
        cfg["anomaly"]["ml"].update({"enabled": True, "model_path": None, "warmup_tracks": 15})
        eng = SceneEngine(cfg, (W, H), AnomalyModel(cfg["anomaly"]["ml"]))
        t_base, frame = 0.0, 0
        for n in range(20):
            v = rng.uniform(90, 130)
            for i in range(40):
                eng.update(t_base + i / FPS, frame, [det(n, 100 + v * i / FPS, 300 + rng.normal(0, 0.3))])
                frame += 1
            t_base += 4.0 + 3.5
            eng.update(t_base, frame, [])                      # lets the finished track time out
            frame += 1
        self.assertTrue(eng.ml.is_fitted)


if __name__ == "__main__":
    unittest.main()
