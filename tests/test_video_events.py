import numpy as np
import pandas as pd

from tc4.video.analyzer import VideoConfig, detect_video_events, events_to_findings
from tc4.video.features import frame_features, joint_angle


def _kps(arm_angle_deg: float) -> np.ndarray:
    """Pessoa em pé; braço esquerdo e direito elevados ao mesmo ângulo."""
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    k[5] = [100, 100, 0.9]  # ombro esq
    k[6] = [160, 100, 0.9]  # ombro dir
    k[11] = [105, 220, 0.9]
    k[12] = [155, 220, 0.9]
    a = np.radians(arm_angle_deg)
    k[7] = [100 - 60 * np.sin(a), 100 + 60 * np.cos(a), 0.9]
    k[8] = [160 + 60 * np.sin(a), 100 + 60 * np.cos(a), 0.9]
    k[9] = k[7] + [0, 0, 0]
    k[10] = k[8] + [0, 0, 0]
    return k


def test_joint_angle_basic():
    assert abs(joint_angle(np.array([0, 1]), np.array([0, 0]), np.array([1, 0])) - 90) < 1e-6


def test_shoulder_angle_from_keypoints():
    f = frame_features(_kps(90), (80, 80, 180, 400), 480)
    assert abs(f["shoulder_l"] - 90) < 6 and abs(f["shoulder_r"] - 90) < 6
    assert f["trunk_incl"] < 5


def _session(amplitudes, fps=10.0):
    rows, t = [], 0.0
    for amp in amplitudes:
        for phase in np.linspace(0, np.pi, int(fps * 2)):
            f = frame_features(_kps(20 + (amp - 20) * np.sin(phase)), (80, 80, 180, 400), 480)
            rows.append({"t": t, "person": True, "n_persons": 1, "speed": 0.1, **f})
            t += 1 / fps
    return pd.DataFrame(rows)


def test_reduced_amplitude_repetition_is_flagged():
    df = _session([150, 150, 150, 80, 150, 150])
    events, summary = detect_video_events(df, 10.0, VideoConfig())
    assert summary["repetitions"] == 6
    reduced = [e for e in events if e["type"] == "amplitude_reduzida"]
    assert len(reduced) == 1 and reduced[0]["metrics"]["repetition"] == 4


def test_progressive_fatigue_and_fall():
    df = _session([160, 155, 150, 120, 110, 100, 95])
    n = 20
    fall = df.iloc[-1:].copy()
    rows = []
    for i in range(n):
        r = fall.iloc[0].to_dict()
        r["t"] = df["t"].iloc[-1] + (i + 1) * 0.1
        r["hip_y"] = 0.46 + (0.45 if i >= 3 else 0.0)
        r["bbox_aspect"] = 2.5 if i >= 3 else 0.3
        rows.append(r)
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    events, _ = detect_video_events(df, 10.0, VideoConfig())
    types = {e["type"] for e in events}
    assert "fadiga_progressiva" in types
    assert "possivel_queda" in types
    findings = events_to_findings(events, "SYN-T")
    assert any("queda" in f.tags and f.severity == 3 for f in findings)
