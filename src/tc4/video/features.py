"""Cálculos biomecânicos a partir de keypoints (formato COCO-17 do YOLOv8-pose).

Este módulo usa apenas NumPy/Pandas/SciPy para poder ser testado sem modelos de visão.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KP = {
    "nose": 0, "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14,
    "l_ankle": 15, "r_ankle": 16,
}
FEATURES = ["shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "trunk_incl", "knee_l",
            "knee_r", "asymmetry", "bbox_aspect", "hip_y", "speed"]


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ângulo em graus no vértice b formado pelos segmentos b->a e b->c."""
    v1, v2 = np.asarray(a, float) - b, np.asarray(c, float) - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def frame_features(kps: np.ndarray, bbox: tuple[float, float, float, float],
                   frame_h: float, conf_thr: float = 0.3) -> dict:
    """kps: array (17, 3) com x, y, confiança. Retorna ângulos e medidas do quadro.

    - shoulder_*: elevação do braço (quadril–ombro–cotovelo): ~0° braço ao lado do corpo,
      ~90° na horizontal, ~180° acima da cabeça.
    - trunk_incl: inclinação do tronco em relação à vertical (compensação).
    """
    kps = np.asarray(kps, float)

    def p(name: str) -> np.ndarray | None:
        x, y, c = kps[KP[name]]
        return np.array([x, y]) if c >= conf_thr else None

    def ang(a: str, b: str, c: str) -> float:
        pa, pb, pc = p(a), p(b), p(c)
        return joint_angle(pa, pb, pc) if pa is not None and pb is not None and pc is not None \
            else float("nan")

    out = {
        "shoulder_l": ang("l_hip", "l_shoulder", "l_elbow"),
        "shoulder_r": ang("r_hip", "r_shoulder", "r_elbow"),
        "elbow_l": ang("l_shoulder", "l_elbow", "l_wrist"),
        "elbow_r": ang("r_shoulder", "r_elbow", "r_wrist"),
        "knee_l": ang("l_hip", "l_knee", "l_ankle"),
        "knee_r": ang("r_hip", "r_knee", "r_ankle"),
    }
    ls, rs, lh, rh = p("l_shoulder"), p("r_shoulder"), p("l_hip"), p("r_hip")
    if all(v is not None for v in (ls, rs, lh, rh)):
        mid_s, mid_h = (ls + rs) / 2, (lh + rh) / 2
        out["trunk_incl"] = joint_angle(mid_h + np.array([0.0, -100.0]), mid_h, mid_s)
        out["hip_y"] = float(mid_h[1] / frame_h)
    else:
        out["trunk_incl"] = float("nan")
        out["hip_y"] = float("nan")
    out["asymmetry"] = abs(out["shoulder_l"] - out["shoulder_r"])
    x1, y1, x2, y2 = bbox
    out["bbox_aspect"] = float((x2 - x1) / max(y2 - y1, 1e-6))
    out["bbox_h"] = float(y2 - y1)
    return out


def add_speed(df: pd.DataFrame, kps_seq: list[np.ndarray | None], fps_eff: float) -> pd.DataFrame:
    """Velocidade média dos keypoints (altura do corpo por segundo)."""
    speeds = [np.nan]
    for prev, cur, h in zip(kps_seq[:-1], kps_seq[1:], df["bbox_h"].to_numpy()[1:]):
        if prev is None or cur is None or not h:
            speeds.append(np.nan)
            continue
        ok = (prev[:, 2] > 0.3) & (cur[:, 2] > 0.3)
        if ok.sum() < 4:
            speeds.append(np.nan)
            continue
        disp = np.linalg.norm(cur[ok, :2] - prev[ok, :2], axis=1).mean()
        speeds.append(float(disp / h * fps_eff))
    df = df.copy()
    df["speed"] = speeds[: len(df)]
    return df


def smooth(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    cols = [c for c in FEATURES if c in df.columns]
    out = df.copy()
    out[cols] = df[cols].interpolate(limit=window, limit_direction="both") \
        .rolling(window, center=True, min_periods=1).median()
    return out
