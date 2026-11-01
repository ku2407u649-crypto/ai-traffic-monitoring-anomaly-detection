"""ML-based anomaly detection on vehicle motion features (Isolation Forest).

The model learns what *normal* motion looks like (speed profile, smoothness,
straightness of the path) and gives a low score to tracks that behave very
differently, for example sudden braking, erratic swerving or U-turns.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ["mean_speed", "std_speed", "max_accel", "heading_change", "straightness", "duration"]


def track_features(pts: list[tuple[float, float, float]], diag: float) -> np.ndarray | None:
    """pts = [(t, x, y), ...]. Returns a feature vector or None if too short."""
    if len(pts) < 8:
        return None
    a = np.asarray(pts, dtype=float)
    t, xy = a[:, 0], a[:, 1:3]
    dt = np.diff(t)
    keep = dt > 1e-6
    if keep.sum() < 6:
        return None
    step = np.diff(xy, axis=0)[keep]
    dt = dt[keep]
    dist = np.hypot(step[:, 0], step[:, 1])
    v = dist / dt / diag                                   # normalised speed per second
    k = min(5, len(v))
    v_s = np.convolve(v, np.ones(k) / k, mode="valid") if len(v) >= k else v
    acc = np.abs(np.diff(v_s)) / np.maximum(dt[: len(v_s) - 1], 1e-3) if len(v_s) > 1 else np.array([0.0])
    moving = dist > 0.5
    if moving.sum() >= 3:
        ang = np.arctan2(step[moving, 1], step[moving, 0])
        d_ang = np.abs((np.diff(ang) + math.pi) % (2 * math.pi) - math.pi)
        heading = float(np.mean(d_ang))
    else:
        heading = 0.0
    path = float(dist.sum())
    net = float(np.hypot(*(xy[-1] - xy[0])))
    straight = net / path if path > 1e-6 else 1.0
    return np.array([float(v_s.mean()), float(v_s.std()), float(np.clip(acc.max(), 0, 5)),
                     heading, straight, float(t[-1] - t[0])])


class AnomalyModel:
    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", True))
        self.warmup = int(cfg.get("warmup_tracks", 40))
        self.contamination = float(cfg.get("contamination", 0.03))
        self.threshold = float(cfg.get("score_threshold", -0.02))
        self.samples: list[np.ndarray] = []
        self.model = None
        path = cfg.get("model_path")
        if self.enabled and path:
            p = Path(path)
            if not p.is_absolute():
                p = Path(__file__).resolve().parent.parent / p
            if p.exists():
                self.load(p)

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def add_sample(self, feat: np.ndarray) -> None:
        if not self.enabled or self.is_fitted:
            return
        self.samples.append(feat)
        if len(self.samples) >= self.warmup:
            self.fit()

    def fit(self, X: np.ndarray | None = None) -> None:
        X = np.vstack(self.samples) if X is None else X
        self.model = make_pipeline(
            StandardScaler(),
            IsolationForest(n_estimators=200, contamination=self.contamination, random_state=0),
        ).fit(X)

    def score(self, feat: np.ndarray) -> tuple[bool, float]:
        """Return (is_anomaly, score). Lower score = more unusual."""
        if not self.is_fitted:
            return False, 0.0
        s = float(self.model.decision_function(feat.reshape(1, -1))[0])
        return s < self.threshold, s

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
