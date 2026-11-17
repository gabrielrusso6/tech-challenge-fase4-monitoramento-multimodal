"""Padrões de movimentação do paciente durante a internação.

Entrada: série de contagens de atividade (ex.: acelerômetro de pulseira ou sensor de leito)
agregadas a cada 10 minutos. O repositório inclui um gerador SINTÉTICO com eventos rotulados,
pois não há dataset público equivalente com internação e quedas anotadas.

Detecções:
- imobilidade prolongada no período diurno (risco de lesão por pressão / sedação excessiva);
- agitação noturna (atividade muito acima do perfil esperado para a hora);
- possível queda: pico extremo seguido de imobilidade;
- declínio funcional: tendência de queda na atividade diária total.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..models import ATENCAO, CRITICO, Finding
from .vitals import keep_runs, mask_segments

STEP_MIN = 10


def synthetic_activity(patient_id: str = "SYN-101", days: int = 5, seed: int = 7,
                       start: str = "2026-11-10 00:00", inject: bool = True
                       ) -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(seed)
    steps_per_day = 24 * 60 // STEP_MIN
    n = days * steps_per_day
    ts = pd.date_range(start, periods=n, freq=f"{STEP_MIN}min")
    hours = ts.hour + ts.minute / 60
    day_index = np.arange(n) // steps_per_day
    awake = ((hours >= 7) & (hours < 22)).astype(float)
    recovery = 0.6 + 0.12 * day_index  # paciente melhora ao longo da internação
    base = awake * (40 + 25 * np.sin((hours - 7) / 15 * np.pi)) * recovery + 3
    activity = rng.poisson(np.clip(base, 0.5, None)).astype(float)
    events: list[dict] = []
    if inject:
        def span(day: int, h0: float, h1: float) -> slice:
            a = day * steps_per_day + int(h0 * 60 / STEP_MIN)
            b = day * steps_per_day + int(h1 * 60 / STEP_MIN)
            return slice(a, b)

        s = span(2, 10, 14)
        activity[s] = rng.poisson(0.3, s.stop - s.start)
        events.append({"kind": "imobilidade", "start": str(ts[s.start]), "end": str(ts[s.stop])})
        s = span(3, 2, 4)
        activity[s] = rng.poisson(60, s.stop - s.start)
        events.append({"kind": "agitacao_noturna", "start": str(ts[s.start]),
                       "end": str(ts[s.stop])})
        s = span(4, 7.5, 8.5)
        activity[s.start] = 260
        activity[s.start + 1:s.stop] = rng.poisson(0.2, s.stop - s.start - 1)
        events.append({"kind": "queda", "start": str(ts[s.start]), "end": str(ts[s.stop])})
    df = pd.DataFrame({"patient_id": patient_id, "timestamp": ts, "activity": activity})
    return df, events


def load_activity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _expected_profile(df: pd.DataFrame, baseline_days: int = 2) -> pd.Series:
    start = df["timestamp"].min().normalize()
    base = df[df["timestamp"] < start + pd.Timedelta(days=baseline_days)]
    slot = base["timestamp"].dt.hour
    return base.groupby(slot)["activity"].median()


def analyze_activity(df: pd.DataFrame, patient_id: str | None = None,
                     immobility_min: int = 120, night_z: float = 3.0) -> list[Finding]:
    patient_id = patient_id or str(df["patient_id"].iloc[0])
    findings: list[Finding] = []
    ts = df["timestamp"]
    act = df["activity"].to_numpy(dtype=float)
    hours = ts.dt.hour.to_numpy()
    daytime = (hours >= 8) & (hours < 20)
    night = (hours >= 23) | (hours < 6)

    # Queda: pico extremo seguido de >= 30 min quase sem movimento.
    p99 = np.percentile(act, 99.5)
    fall_idx = set()
    after = int(30 / STEP_MIN)
    for i in np.flatnonzero(act > max(p99 * 1.3, 120)):
        nxt = act[i + 1:i + 1 + after]
        if len(nxt) == after and nxt.max() <= 2:
            fall_idx.update(range(i, i + 1 + after))
            findings.append(Finding(
                patient_id=patient_id, source="movement", kind="possivel_queda",
                severity=CRITICO,
                message="Pico de movimento extremo seguido de imobilidade: possível queda",
                offset=ts.iloc[i].strftime("%Y-%m-%d %H:%M"), tags=["queda", "mobilidade"],
                evidence={"peak": float(act[i]), "threshold": round(float(p99 * 1.3), 1)}))

    # Imobilidade prolongada no período diurno.
    still = daytime & (act <= 2)
    still[list(fall_idx)] = False
    min_steps = immobility_min // STEP_MIN
    for s, e in mask_segments(keep_runs(still, min_steps), merge_gap=0):
        minutes = (e - s) * STEP_MIN
        findings.append(Finding(
            patient_id=patient_id, source="movement", kind="imobilidade_prolongada",
            severity=ATENCAO,
            message=f"Imobilidade diurna por {minutes} min (risco de lesão por pressão, "
                    "sedação ou piora clínica)",
            offset=f"{ts.iloc[s]:%Y-%m-%d %H:%M}–{ts.iloc[e - 1]:%H:%M}",
            tags=["imobilidade", "mobilidade", "fadiga"], evidence={"minutes": minutes}))

    # Agitação noturna comparada ao perfil esperado por hora (linha de base dos 2 primeiros dias).
    profile = _expected_profile(df)
    expected = pd.Series(hours).map(profile).fillna(np.median(act)).to_numpy()
    resid = act - expected
    mad = np.median(np.abs(resid - np.median(resid))) or 1.0
    z = resid / (1.4826 * mad)
    agitated = night & (z > night_z)
    agitated[list(fall_idx)] = False
    for s, e in mask_segments(keep_runs(agitated, 3), merge_gap=1):
        findings.append(Finding(
            patient_id=patient_id, source="movement", kind="agitacao_noturna", severity=ATENCAO,
            message=f"Agitação noturna: atividade muito acima do esperado por "
                    f"{(e - s) * STEP_MIN} min (dor, delirium ou desconforto?)",
            offset=f"{ts.iloc[s]:%Y-%m-%d %H:%M}–{ts.iloc[e - 1]:%H:%M}",
            score=float(z[s:e].max()), tags=["agitacao", "mobilidade"],
            evidence={"max_z": round(float(z[s:e].max()), 1)}))

    # Declínio funcional: inclinação da atividade diária total.
    daily = df.groupby(ts.dt.date)["activity"].sum()
    if len(daily) >= 3:
        x = np.arange(len(daily))
        slope = np.polyfit(x, daily.to_numpy(dtype=float), 1)[0]
        rel = slope / max(daily.mean(), 1.0)
        if rel < -0.15:
            findings.append(Finding(
                patient_id=patient_id, source="movement", kind="declinio_funcional",
                severity=ATENCAO,
                message=f"Atividade diária em queda ({rel:.0%} por dia)",
                offset=str(daily.index[-1]), tags=["fadiga", "mobilidade"],
                evidence={"relative_slope_per_day": round(float(rel), 3)}))
    return findings


def plot_activity(df: pd.DataFrame, findings: list[Finding], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(df["timestamp"], df["activity"], lw=0.7, color="#1f4e79")
    for f in findings:
        if f.kind == "declinio_funcional":
            continue
        start = pd.Timestamp(f.offset.split("–")[0])
        ax.axvline(start, color="#c00000" if f.severity == CRITICO else "#e69500", lw=1.2)
        ax.text(start, ax.get_ylim()[1] * 0.92, f.kind, rotation=90, fontsize=7, va="top")
    ax.set_ylabel("atividade / 10 min")
    ax.set_title("Movimentação durante a internação (sintético) e anomalias")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
