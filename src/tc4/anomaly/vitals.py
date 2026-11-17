"""Detecção de anomalias em séries temporais de sinais vitais.

Fonte real: PhysioNet BIDMC PPG and Respiration Dataset (53 registros de 8 min, 1 Hz, com
FC, frequência respiratória e SpO2; licença ODC-By 1.0). O BIDMC NÃO tem pressão arterial:
as colunas `sbp`/`dbp` são SINTÉTICAS, correlacionadas à FC, e isso é declarado no relatório.

Três detectores complementares:
1. Regras clínicas com persistência (limites educacionais, não protocolo do hospital);
2. Z-score robusto causal (mediana/MAD de uma janela móvel) por sinal;
3. Isolation Forest multivariado treinado no período inicial (linha de base) do paciente.

`OnlineVitalsMonitor` aplica os mesmos princípios amostra a amostra (simulação de tempo real).
"""
from __future__ import annotations

import math
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from ..models import ATENCAO, CRITICO, Finding

VITALS = ["hr", "resp", "spo2", "sbp", "dbp"]
UNITS = {"hr": "bpm", "resp": "irpm", "spo2": "%", "sbp": "mmHg", "dbp": "mmHg"}
NAMES = {
    "hr": "Frequência cardíaca",
    "resp": "Frequência respiratória",
    "spo2": "SpO2",
    "sbp": "PA sistólica",
    "dbp": "PA diastólica",
}
GROUP = {"hr": "hemodinamico", "sbp": "hemodinamico", "dbp": "hemodinamico",
         "resp": "respiratorio", "spo2": "respiratorio"}
# Limites EDUCACIONAIS para adulto internado (não substituem protocolo institucional).
CLINICAL_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "hr": (45.0, 120.0),
    "resp": (8.0, 25.0),
    "spo2": (92.0, None),
    "sbp": (90.0, 160.0),
    "dbp": (50.0, 100.0),
}
MAD_FLOOR = {"hr": 1.5, "resp": 1.0, "spo2": 0.7, "sbp": 2.0, "dbp": 1.5}
EVENT_KINDS = ["taquicardia", "dessaturacao", "hipotensao", "taquipneia", "pico_artefato"]
BIDMC_URL = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/bidmc_{n:02d}_Numerics.csv"


# --------------------------------------------------------------------------- dados

def download_bidmc(records: list[int], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for n in records:
        path = dest / f"bidmc_{n:02d}_Numerics.csv"
        if not path.exists():
            urllib.request.urlretrieve(BIDMC_URL.format(n=n), path)
        paths.append(path)
    return paths


def _ar_noise(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + rng.normal(0, sigma)
    return out


def add_synthetic_blood_pressure(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Acrescenta PA sistólica/diastólica SINTÉTICAS (o BIDMC não possui PA)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    drift = _ar_noise(rng, n, 0.97, 0.5)
    hr_dev = (df["hr"] - df["hr"].median()).to_numpy()
    out = df.copy()
    out["sbp"] = 118 + 0.3 * hr_dev + 2.0 * drift + rng.normal(0, 1.0, n)
    out["dbp"] = 74 + 0.15 * hr_dev + 1.2 * drift + rng.normal(0, 0.8, n)
    return out


def load_bidmc_numerics(path: Path, seed: int = 0) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    def col(*names: str) -> str | None:
        for name in names:
            for c in raw.columns:
                if c == name or c.startswith(name):
                    return c
        return None

    time_col = col("time")
    df = pd.DataFrame({"t": raw[time_col].astype(float) if time_col else np.arange(len(raw))})
    df["t"] = df["t"] - df["t"].iloc[0]
    rng = np.random.default_rng(seed)
    defaults = {"hr": 80.0, "resp": 16.0, "spo2": 97.0}
    for name in ("hr", "resp", "spo2"):
        c = col(name)
        values = pd.to_numeric(raw[c], errors="coerce") if c else pd.Series(np.nan, index=raw.index)
        values = values.where(values > 0)  # zeros no BIDMC indicam ausência de leitura
        if values.notna().sum() < len(values) * 0.5:
            # Sinal ausente no registro: preenche com linha de base sintética (documentado).
            values = pd.Series(defaults[name] + rng.normal(0, 0.5, len(raw)), index=raw.index)
        df[name] = values.astype(float).interpolate(limit_direction="both").to_numpy(dtype=float)
    df["spo2"] = df["spo2"].clip(upper=100)
    return add_synthetic_blood_pressure(df, seed=seed)


def synthetic_vitals(n: int = 480, seed: int = 0) -> pd.DataFrame:
    """Série sintética estável (usada em testes e quando não há internet)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"t": np.arange(n, dtype=float)})
    df["hr"] = 78 + _ar_noise(rng, n, 0.95, 0.8) + rng.normal(0, 0.5, n)
    df["resp"] = 16 + _ar_noise(rng, n, 0.9, 0.3) + rng.normal(0, 0.3, n)
    df["spo2"] = np.clip(97.5 + _ar_noise(rng, n, 0.9, 0.15), 0, 100)
    return add_synthetic_blood_pressure(df, seed=seed)


def inject_anomalies(df: pd.DataFrame, kinds: list[str], seed: int = 0, warmup: int = 150
                     ) -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
    """Injeta eventos rotulados e retorna (série, rótulos por segundo, eventos)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in ("hr", "resp", "spo2", "sbp", "dbp"):
        if c in out.columns:  # dados reais (BIDMC) vêm como inteiros
            out[c] = out[c].astype(float)
    n = len(out)
    labels = np.zeros(n, dtype=bool)
    events: list[dict] = []
    usable = n - warmup - 20
    slot = usable // max(1, len(kinds))
    for i, kind in enumerate(kinds):
        duration = 3 if kind == "pico_artefato" else int(rng.integers(45, 76))
        lo = warmup + i * slot
        hi = max(lo + 1, lo + slot - duration - 10)
        start = int(rng.integers(lo, hi))
        end = min(n, start + duration)
        idx = np.arange(start, end)
        k = np.arange(len(idx))
        profile = np.clip(np.minimum((k + 1) / 12.0, (len(idx) - k) / 8.0), 0, 1)
        if kind == "taquicardia":
            out.loc[idx, "hr"] += 50 * profile
            vitals = ["hr"]
        elif kind == "dessaturacao":
            out.loc[idx, "spo2"] -= 9 * profile
            vitals = ["spo2"]
        elif kind == "hipotensao":
            out.loc[idx, "sbp"] -= 40 * profile
            out.loc[idx, "dbp"] -= 22 * profile
            vitals = ["sbp", "dbp"]
        elif kind == "taquipneia":
            out.loc[idx, "resp"] += 14 * profile
            vitals = ["resp"]
        elif kind == "pico_artefato":
            out.loc[idx, "hr"] += 70
            vitals = ["hr"]
        else:
            raise ValueError(f"tipo de evento desconhecido: {kind}")
        labels[start:end] = True
        events.append({"kind": kind, "start": start, "end": end, "vitals": vitals})
    out["spo2"] = out["spo2"].clip(upper=100)
    return out, labels, events


# --------------------------------------------------------------------------- utilidades

def keep_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Mantém apenas sequências de True com comprimento >= min_len (persistência)."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    n, i = len(mask), 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def mask_segments(mask: np.ndarray, merge_gap: int = 5) -> list[tuple[int, int]]:
    """Converte máscara em segmentos [início, fim), unindo lacunas pequenas."""
    segments: list[list[int]] = []
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    for i in idx:
        if segments and i - segments[-1][1] <= merge_gap:
            segments[-1][1] = int(i) + 1
        else:
            segments.append([int(i), int(i) + 1])
    return [(s, e) for s, e in segments]


def _breach(values: pd.Series, vital: str) -> pd.Series:
    lo, hi = CLINICAL_LIMITS[vital]
    cond = pd.Series(False, index=values.index)
    if lo is not None:
        cond |= values < lo
    if hi is not None:
        cond |= values > hi
    return cond


def robust_z(series: pd.Series, window: int = 120, min_periods: int = 30,
             floor: float = 1.0) -> pd.Series:
    """Z-score robusto causal: compara cada ponto com a mediana/MAD dos pontos anteriores."""
    base = series.shift(1)
    med = base.rolling(window, min_periods=min_periods).median()
    mad = (base - med).abs().rolling(window, min_periods=min_periods).median()
    scale = 1.4826 * mad.clip(lower=floor)
    return ((series - med) / scale).fillna(0.0)


def _iforest_features(df: pd.DataFrame) -> np.ndarray:
    values = df[VITALS]
    diffs = values.diff(5).fillna(0.0).add_suffix("_d5")
    return pd.concat([values, diffs], axis=1).to_numpy(dtype=float)


def iforest_scores(df: pd.DataFrame, train_seconds: int = 120, seed: int = 0
                   ) -> tuple[np.ndarray, float]:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler

    X = _iforest_features(df)
    train = X[:train_seconds]
    scaler = RobustScaler().fit(train)
    model = IsolationForest(n_estimators=200, random_state=seed).fit(scaler.transform(train))
    scores = -model.score_samples(scaler.transform(X))
    threshold = float(np.quantile(scores[:train_seconds], 0.995)) + 0.05
    return scores, threshold


# --------------------------------------------------------------------------- detecção em lote

def detect_batch(df: pd.DataFrame, persistence: int = 5, z_threshold: float = 4.0,
                 train_seconds: int = 120, seed: int = 0) -> pd.DataFrame:
    """Retorna DataFrame booleano com colunas rule, zscore, iforest e ensemble."""
    out = pd.DataFrame(index=df.index)
    rule_any = np.zeros(len(df), dtype=bool)
    z_any = np.zeros(len(df), dtype=bool)
    for v in VITALS:
        rule_v = keep_runs(_breach(df[v], v).to_numpy(), persistence)
        out[f"rule_{v}"] = rule_v
        rule_any |= rule_v
        z = robust_z(df[v], floor=MAD_FLOOR[v])
        out[f"z_{v}"] = z.to_numpy()
        z_any |= np.abs(z.to_numpy()) >= z_threshold
    out["rule"] = rule_any
    out["zscore"] = keep_runs(z_any, 3)
    scores, thr = iforest_scores(df, train_seconds=train_seconds, seed=seed)
    out["iforest_score"] = scores
    out["iforest"] = keep_runs(scores > thr, 10)
    out["ensemble"] = out["rule"] | out["zscore"] | out["iforest"]
    return out


def evaluate_flags(flags: np.ndarray, labels: np.ndarray, events: list[dict],
                   tolerance: int = 10) -> dict:
    flags = np.asarray(flags, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    tp = int((flags & labels).sum())
    fp = int((flags & ~labels).sum())
    fn = int((~flags & labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    detected, latencies = 0, []
    covered = np.zeros_like(flags)
    for ev in events:
        s, e = ev["start"], min(len(flags), ev["end"] + tolerance)
        covered[max(0, s - tolerance):e] = True
        window = flags[s:e]
        if window.any():
            detected += 1
            latencies.append(int(np.argmax(window)))
    false_segments = [seg for seg in mask_segments(flags) if not covered[seg[0]:seg[1]].any()]
    hours = len(flags) / 3600.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "event_recall": round(detected / len(events), 4) if events else float("nan"),
        "mean_latency_s": round(float(np.mean(latencies)), 2) if latencies else float("nan"),
        "false_alarms": len(false_segments),
        "false_alarms_per_hour": round(len(false_segments) / hours, 2) if hours else 0.0,
    }


def kinds_for_record(index: int) -> list[str]:
    return [EVENT_KINDS[(2 * index) % 5], EVENT_KINDS[(2 * index + 1) % 5]]


def evaluate_records(frames: dict[str, pd.DataFrame], seed: int = 42
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia cada método em cada registro com anomalias injetadas e rotuladas."""
    rows = []
    for i, (name, df) in enumerate(frames.items()):
        injected, labels, events = inject_anomalies(df, kinds_for_record(i), seed=seed + i)
        det = detect_batch(injected, seed=seed)
        for method in ("rule", "zscore", "iforest", "ensemble"):
            metrics = evaluate_flags(det[method].to_numpy(), labels, events)
            rows.append({"record": name, "method": method,
                         "events": ",".join(e["kind"] for e in events), **metrics})
    detail = pd.DataFrame(rows)
    summary = (detail.groupby("method")[["precision", "recall", "f1", "event_recall",
                                          "mean_latency_s", "false_alarms_per_hour"]]
               .mean().round(3).reset_index())
    return detail, summary


def segments_to_findings(df: pd.DataFrame, det: pd.DataFrame, patient_id: str,
                         merge_gap: int = 5) -> list[Finding]:
    findings = []
    for s, e in mask_segments(det["ensemble"].to_numpy(), merge_gap=merge_gap):
        sub = df.iloc[s:e]
        dsub = det.iloc[s:e]
        involved = []
        details = []
        for v in VITALS:
            hit_rule = bool(dsub[f"rule_{v}"].any())
            hit_z = bool((dsub[f"z_{v}"].abs() >= 4.0).any())
            if hit_rule or hit_z:
                involved.append(v)
                lo, hi = CLINICAL_LIMITS[v]
                if lo is not None and sub[v].min() < lo:
                    extreme = sub[v].min()
                elif hi is not None and sub[v].max() > hi:
                    extreme = sub[v].max()
                else:
                    zs = np.abs(dsub[f"z_{v}"].to_numpy())
                    extreme = sub[v].iloc[int(np.argmax(zs))]
                details.append(f"{NAMES[v]} {extreme:.0f} {UNITS[v]}")
        severity = CRITICO if bool(dsub["rule"].any()) else ATENCAO
        method = [m for m in ("rule", "zscore", "iforest") if bool(dsub[m].any())]
        tags = sorted({GROUP[v] for v in involved} | set(involved))
        t0, t1 = float(sub["t"].iloc[0]), float(sub["t"].iloc[-1])
        msg = ("Anomalia em sinais vitais: " + ("; ".join(details) if details
               else "padrão multivariado atípico (Isolation Forest)"))
        findings.append(Finding(
            patient_id=patient_id, source="vitals", kind="sinais_vitais", severity=severity,
            message=msg, offset=f"t={t0:.0f}-{t1:.0f}s",
            score=float(dsub["iforest_score"].max()), tags=tags,
            evidence={"methods": method, "vitals": involved, "start_s": t0, "end_s": t1},
        ))
    return findings


def plot_vitals(df: pd.DataFrame, det: pd.DataFrame, events: list[dict], path: Path,
                title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(VITALS), 1, figsize=(12, 11), sharex=True)
    t = df["t"].to_numpy()
    flags = det["ensemble"].to_numpy()
    for ax, v in zip(axes, VITALS):
        ax.plot(t, df[v], lw=0.9, color="#1f4e79")
        lo, hi = CLINICAL_LIMITS[v]
        for lim in (lo, hi):
            if lim is not None:
                ax.axhline(lim, color="#c00000", lw=0.7, ls="--")
        for ev in events:
            ax.axvspan(t[ev["start"]], t[min(len(t) - 1, ev["end"] - 1)], color="#999999",
                       alpha=0.25)
        for s, e in mask_segments(flags):
            ax.axvspan(t[s], t[e - 1], color="#ff4d4d", alpha=0.25)
        suffix = " (sintética)" if v in ("sbp", "dbp") else ""
        ax.set_ylabel(f"{v.upper()}{suffix}\n{UNITS[v]}", fontsize=8)
    axes[-1].set_xlabel("tempo (s)  —  cinza: anomalia injetada | vermelho: detectada")
    fig.suptitle(title or "Sinais vitais e anomalias detectadas")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- tempo real

class OnlineVitalsMonitor:
    """Detector incremental: recebe uma amostra por segundo e devolve achados novos.

    - limite clínico mantido por `persistence` s  -> CRÍTICO;
    - |z| robusto >= `z_threshold` por 3 s         -> ATENÇÃO;
    - Isolation Forest (após linha de base) por 10 s -> ATENÇÃO.
    Cada tipo de achado tem `cooldown` para evitar alertas repetidos.
    """

    def __init__(self, patient_id: str, window: int = 120, z_threshold: float = 4.0,
                 persistence: int = 5, cooldown: int = 60, baseline: int = 120,
                 use_iforest: bool = True, seed: int = 0) -> None:
        self.patient_id = patient_id
        self.z_threshold = z_threshold
        self.persistence = persistence
        self.cooldown = cooldown
        self.baseline = baseline
        self.use_iforest = use_iforest
        self.seed = seed
        self.buf = {v: deque(maxlen=window) for v in VITALS}
        self.rule_run = {v: 0 for v in VITALS}
        self.z_run = {v: 0 for v in VITALS}
        self.if_run = 0
        self._rows: list[list[float]] = []
        self._model = None
        self._scaler = None
        self._if_thr = math.inf
        self._last: dict[str, float] = {}
        self._prev: deque[list[float]] = deque(maxlen=6)
        self.t = 0.0

    def _ok(self, key: str, t: float) -> bool:
        last = self._last.get(key)
        if last is not None and t - last < self.cooldown:
            return False
        self._last[key] = t
        return True

    def _finding(self, v: str, x: float, z: float, t: float, severity: int, reason: str
                 ) -> Finding:
        return Finding(
            patient_id=self.patient_id, source="vitals", kind=f"{v}_{reason.split()[0]}",
            severity=severity,
            message=f"{NAMES[v]} = {x:.0f} {UNITS[v]} ({reason}; z={z:+.1f})",
            offset=f"t={t:.0f}s", score=abs(z), tags=[v, GROUP[v]],
            evidence={"value": round(x, 2), "z": round(z, 2), "limits": CLINICAL_LIMITS[v]},
        )

    def update(self, sample: dict) -> list[Finding]:
        t = float(sample.get("t", self.t))
        self.t = t + 1
        found: list[Finding] = []
        row = []
        for v in VITALS:
            raw = sample.get(v)
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                row.append(np.nan)
                continue
            x = float(raw)
            row.append(x)
            lo, hi = CLINICAL_LIMITS[v]
            breach = (lo is not None and x < lo) or (hi is not None and x > hi)
            self.rule_run[v] = self.rule_run[v] + 1 if breach else 0
            z = 0.0
            buf = self.buf[v]
            if len(buf) >= 30:
                arr = np.fromiter(buf, dtype=float)
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med)))
                z = (x - med) / (1.4826 * max(mad, MAD_FLOOR[v]))
            self.z_run[v] = self.z_run[v] + 1 if abs(z) >= self.z_threshold else 0
            buf.append(x)
            if self.rule_run[v] == self.persistence and self._ok(f"{v}_limite", t):
                found.append(self._finding(v, x, z, t, CRITICO, "limite clínico ultrapassado"))
            elif (self.z_run[v] == 3 and self.rule_run[v] < self.persistence
                  and self._ok(f"{v}_desvio", t)):
                found.append(self._finding(v, x, z, t, ATENCAO, "desvio estatístico"))
        if self.use_iforest and not any(np.isnan(row)):
            found.extend(self._iforest_step(row, t))
        return found

    def _iforest_step(self, row: list[float], t: float) -> list[Finding]:
        prev = self._prev[0] if len(self._prev) == self._prev.maxlen else row
        feat = row + [a - b for a, b in zip(row, prev)]
        self._prev.append(row)
        if self._model is None:
            self._rows.append(feat)
            if len(self._rows) >= self.baseline:
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import RobustScaler

                X = np.asarray(self._rows)
                self._scaler = RobustScaler().fit(X)
                self._model = IsolationForest(n_estimators=150, random_state=self.seed)
                self._model.fit(self._scaler.transform(X))
                base = -self._model.score_samples(self._scaler.transform(X))
                self._if_thr = float(np.quantile(base, 0.995)) + 0.05
            return []
        score = float(-self._model.score_samples(self._scaler.transform([feat]))[0])
        self.if_run = self.if_run + 1 if score > self._if_thr else 0
        if self.if_run == 10 and self._ok("multivariado", t):
            return [Finding(
                patient_id=self.patient_id, source="vitals", kind="multivariado",
                severity=ATENCAO,
                message="Padrão multivariado atípico nos sinais vitais (Isolation Forest)",
                offset=f"t={t:.0f}s", score=score, tags=["multivariado"],
                evidence={"score": round(score, 3), "threshold": round(self._if_thr, 3)},
            )]
        return []
