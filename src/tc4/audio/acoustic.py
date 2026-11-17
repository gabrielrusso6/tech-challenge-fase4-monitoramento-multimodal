"""Características acústicas locais (NumPy) para indicadores vocais exploratórios.

Os indicadores (pausas, ritmo, monotonia, queda de energia) são HEURÍSTICOS e servem como
triagem para revisão humana. Não são diagnóstico de fadiga, disartria ou doença respiratória.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def to_wav_16k_mono(src: Path, dst: Path) -> Path:
    """Converte qualquer áudio/vídeo para WAV PCM 16 kHz mono (formato do Azure Speech)."""
    import imageio_ffmpeg

    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                    "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst)], check=True)
    return dst


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y.mean(axis=1), int(sr)


def _frames(y: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(y) < size:
        y = np.pad(y, (0, size - len(y)))
    n = 1 + (len(y) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return y[idx]


def voice_activity(y: np.ndarray, sr: int, frame_ms: int = 30, hop_ms: int = 10
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    """Retorna (máscara de fala por quadro, energia em dB, duração do hop em s)."""
    size, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    fr = _frames(y, size, hop)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)
    thr = lo + 0.35 * (hi - lo)
    mask = db > thr
    # suavização: remove buracos/ilhas menores que 50 ms
    k = max(1, int(50 / hop_ms))
    smoothed = np.convolve(mask.astype(float), np.ones(k) / k, mode="same") > 0.5
    return smoothed, db, hop / sr


def pauses(mask: np.ndarray, hop_s: float, min_pause_s: float = 0.25) -> list[float]:
    """Durações das pausas internas (entre o primeiro e o último trecho de fala)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    inner = ~mask[idx[0]:idx[-1] + 1]
    out, run = [], 0
    for v in inner:
        if v:
            run += 1
        elif run:
            if run * hop_s >= min_pause_s:
                out.append(run * hop_s)
            run = 0
    return out


def pitch_track(y: np.ndarray, sr: int, mask: np.ndarray, hop_s: float,
                fmin: float = 75.0, fmax: float = 400.0, max_frames: int = 1500) -> np.ndarray:
    """F0 por autocorrelação nos quadros com fala."""
    size = int(sr * 0.04)
    hop = int(sr * hop_s)
    voiced = np.flatnonzero(mask)
    if len(voiced) > max_frames:
        voiced = voiced[np.linspace(0, len(voiced) - 1, max_frames).astype(int)]
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    f0 = []
    for i in voiced:
        seg = y[i * hop:i * hop + size]
        if len(seg) < size:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[size - 1:]
        if ac[0] <= 0 or lag_max >= len(ac):
            continue
        lag = lag_min + int(np.argmax(ac[lag_min:lag_max]))
        if ac[lag] / ac[0] > 0.35:  # periodicidade suficiente
            f0.append(sr / lag)
    return np.asarray(f0)


def acoustic_features(path: Path) -> dict:
    y, sr = load_audio(path)
    duration = len(y) / sr
    mask, db, hop_s = voice_activity(y, sr)
    p = pauses(mask, hop_s)
    speech_s = float(mask.sum() * hop_s)
    f0 = pitch_track(y, sr, mask, hop_s)
    voiced_db = db[mask]
    thirds = np.array_split(voiced_db, 3) if len(voiced_db) >= 3 else [voiced_db] * 3
    energy_drop = float(np.mean(thirds[0]) - np.mean(thirds[-1])) if len(voiced_db) >= 3 else 0.0
    minutes = max(duration / 60, 1e-6)
    return {
        "duration_s": round(duration, 2),
        "speech_s": round(speech_s, 2),
        "speech_ratio": round(speech_s / duration, 3) if duration else 0.0,
        "pauses": len(p),
        "pauses_per_min": round(len(p) / minutes, 2),
        "mean_pause_s": round(float(np.mean(p)), 3) if p else 0.0,
        "long_pauses": int(sum(1 for x in p if x >= 1.0)),
        "long_pauses_per_min": round(sum(1 for x in p if x >= 1.0) / minutes, 2),
        "short_pauses_per_min": round(sum(1 for x in p if 0.25 <= x < 0.8) / minutes, 2),
        "f0_mean_hz": round(float(f0.mean()), 1) if len(f0) else None,
        "f0_std_hz": round(float(f0.std()), 1) if len(f0) else None,
        "energy_db_mean": round(float(voiced_db.mean()), 2) if len(voiced_db) else None,
        "energy_drop_db": round(energy_drop, 2),
    }


# Limiares heurísticos (documentados no relatório; calibrar com dados reais).
THRESHOLDS = {
    "speech_rate_wpm_low": 140.0,  # fala conversacional pt-BR ~150-190 palavras/min
    "long_pauses_per_min": 4.0,
    "short_pauses_per_min": 18.0,
    "f0_std_low_hz": 15.0,
    "energy_drop_db": 3.0,
}


def vocal_indicators(features: dict, words_per_min: float | None) -> dict:
    th = THRESHOLDS
    checks = []

    def add(name: str, value, flagged: bool, desc: str, group: str) -> None:
        checks.append({"name": name, "value": value, "flagged": bool(flagged),
                       "description": desc, "group": group})

    if words_per_min is not None:
        add("ritmo_lento", round(words_per_min, 1), words_per_min < th["speech_rate_wpm_low"],
            f"palavras/min < {th['speech_rate_wpm_low']:.0f}", "fadiga")
    add("pausas_longas", features["long_pauses_per_min"],
        features["long_pauses_per_min"] > th["long_pauses_per_min"],
        f"pausas >= 1 s por minuto > {th['long_pauses_per_min']:.0f}", "fadiga")
    if features.get("f0_std_hz") is not None:
        add("monotonia", features["f0_std_hz"], features["f0_std_hz"] < th["f0_std_low_hz"],
            f"variação de F0 < {th['f0_std_low_hz']:.0f} Hz", "fadiga")
    add("queda_energia", features["energy_drop_db"],
        features["energy_drop_db"] > th["energy_drop_db"],
        f"energia cai > {th['energy_drop_db']:.0f} dB do início ao fim", "fadiga")
    add("pausas_respiratorias", features["short_pauses_per_min"],
        features["short_pauses_per_min"] > th["short_pauses_per_min"],
        f"pausas curtas (0,25–0,8 s) por minuto > {th['short_pauses_per_min']:.0f}",
        "respiratorio")
    fatigue = [c for c in checks if c["group"] == "fadiga"]
    fatigue_score = sum(c["flagged"] for c in fatigue) / max(1, len(fatigue))
    return {"checks": checks, "fatigue_score": round(fatigue_score, 2),
            "respiratory_flag": any(c["flagged"] for c in checks if c["group"] == "respiratorio")}
