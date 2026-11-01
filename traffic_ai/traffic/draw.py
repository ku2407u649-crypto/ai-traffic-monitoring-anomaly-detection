"""Drawing helpers for the annotated output video."""
from __future__ import annotations
import cv2
import numpy as np

from .engine import CLASS_NAMES

CLASS_COLORS = {  # BGR
    1: (0, 215, 255), 2: (80, 200, 60), 3: (0, 215, 255), 5: (30, 140, 240), 7: (30, 140, 240), 0: (255, 160, 60),
}
RED = (50, 50, 230)


def _label(img, text, org, color, scale=0.5):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    x = max(0, min(x, img.shape[1] - w - 8))      # keep the label inside the frame
    y = max(y, h + 40)                              # stay below the HUD bar
    cv2.rectangle(img, (x, y - h - 4), (x + w + 6, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_scene(frame, engine, dets, stats, t, fps_text="", banner: str | None = None):
    img = frame.copy()
    # ROI + counting line
    if engine.roi is not None:
        ov = img.copy()
        cv2.fillPoly(ov, [engine.roi.astype(np.int32)], (200, 160, 40))
        img = cv2.addWeighted(ov, 0.10, img, 0.90, 0)
        cv2.polylines(img, [engine.roi.astype(np.int32)], True, (200, 160, 40), 1, cv2.LINE_AA)
    if engine.ped_zone is not None:
        cv2.polylines(img, [engine.ped_zone.astype(np.int32)], True, (255, 160, 60), 2, cv2.LINE_AA)
    if engine.line is not None:
        a, b = engine.line.astype(int)
        cv2.line(img, tuple(a), tuple(b), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "COUNT LINE", (a[0] + 6, a[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # boxes
    for d in dets:
        tid, cls = d["id"], d["cls"]
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        flag = engine.is_flagged(tid, t)
        color = RED if flag else CLASS_COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if flag else 2)
        name = CLASS_NAMES.get(cls, f"c{cls}")
        tr = engine.tracks.get(tid)
        txt = f"{name} #{tid}"
        if tr is not None and tr.speed_kmh is not None and cls != 0:
            txt += f" {tr.speed_kmh:.0f}km/h"
        if flag:
            txt += f" [{flag.upper()}]"
        _label(img, txt, (x1, y1), color)

    # HUD
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 34), (25, 25, 25), -1)
    hud = (f"{fps_text}  t={t:5.1f}s | In frame: {stats['vehicles_in_frame']} | In ROI: {stats['vehicles_in_roi']} "
           f"| Count: {stats['total_count']} | Density: {stats['density']}")
    cv2.putText(img, hud, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if banner:
        cv2.rectangle(img, (0, h - 40), (w, h), (30, 30, 200), -1)
        cv2.putText(img, "ALERT: " + banner, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return img
