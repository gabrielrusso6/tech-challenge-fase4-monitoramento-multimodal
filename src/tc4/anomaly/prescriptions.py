"""Detecção de alterações inesperadas na evolução das prescrições.

Regras explicáveis aplicadas em ordem cronológica, por paciente:
- dose diária acima do máximo de referência (CRÍTICO);
- alteração abrupta da dose diária (>= 2x ou <= 0,5x da anterior) (ATENÇÃO);
- duplicidade de classe sensível (ex.: dois opioides ativos) (ATENÇÃO);
- interação conhecida entre medicamento novo e ativo (severidade da tabela);
- suspensão de uso contínuo (ex.: anticoagulante) sem substituto da mesma classe em 24 h;
- instabilidade terapêutica: 3 ou mais alterações em 24 h, fora da admissão (INFO).

A tabela de referência (data/reference/drug_reference.json) é EDUCACIONAL.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR
from ..models import ATENCAO, CRITICO, INFO, Finding

REFERENCE_PATH = DATA_DIR / "reference" / "drug_reference.json"


def load_reference(path: Path = REFERENCE_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_prescriptions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"patient_id", "timestamp", "drug", "dose_mg", "frequency_per_day", "action"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"colunas ausentes no CSV de prescrições: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["drug"] = df["drug"].str.strip().str.lower()
    df["action"] = df["action"].str.strip().str.lower()
    df["daily_mg"] = df["dose_mg"].astype(float) * df["frequency_per_day"].astype(float)
    return df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


def _f(patient: str, kind: str, severity: int, message: str, ts, tags: list[str],
       evidence: dict) -> Finding:
    return Finding(patient_id=patient, source="prescription", kind=kind, severity=severity,
                   message=message, offset=pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"),
                   tags=tags, evidence=evidence)


def analyze_prescriptions(df: pd.DataFrame, reference: dict | None = None) -> list[Finding]:
    reference = reference or load_reference()
    drugs: dict = reference["drugs"]
    sensitive = set(reference.get("duplicate_sensitive_classes", []))
    interactions = reference.get("interactions", [])
    findings: list[Finding] = []

    for patient, g in df.groupby("patient_id", sort=False):
        active: dict[str, float] = {}
        for row in g.itertuples(index=False):
            drug, ts = row.drug, row.timestamp
            ref = drugs.get(drug, {})
            cls = ref.get("class", "desconhecida")
            tags = [cls] + (["alto_alerta"] if ref.get("high_alert") else [])

            if row.action == "stop":
                active.pop(drug, None)
                if ref.get("continuous"):
                    window = g[(g.timestamp >= ts)
                               & (g.timestamp <= ts + pd.Timedelta(hours=24))
                               & (g.action != "stop") & (g.drug != drug)]
                    replaced = any(drugs.get(d, {}).get("class") == cls for d in window.drug)
                    if not replaced:
                        findings.append(_f(
                            patient, "suspensao_sem_substituto", ATENCAO,
                            f"Suspensão de {drug} ({cls}, uso contínuo) sem substituto da "
                            "mesma classe nas 24 h seguintes", ts, tags,
                            {"drug": drug, "class": cls}))
                continue

            daily = float(row.daily_mg)
            max_daily = ref.get("max_daily_mg")
            if max_daily and daily > max_daily:
                findings.append(_f(
                    patient, "dose_acima_maximo", CRITICO,
                    f"{drug}: dose diária {daily:.0f} mg acima do máximo de referência "
                    f"({max_daily:.0f} mg/dia)", ts, tags,
                    {"drug": drug, "daily_mg": daily, "max_daily_mg": max_daily}))

            previous = active.get(drug)
            if previous is not None and previous > 0 and daily > 0:
                ratio = daily / previous
                if ratio >= 2.0 or ratio <= 0.5:
                    direction = "aumento" if ratio > 1 else "redução"
                    findings.append(_f(
                        patient, "alteracao_abrupta", ATENCAO,
                        f"{drug}: {direction} abrupto da dose diária de {previous:.0f} para "
                        f"{daily:.0f} mg ({ratio:.1f}x)", ts, tags + [direction],
                        {"drug": drug, "from_mg": previous, "to_mg": daily,
                         "ratio": round(ratio, 2)}))
            elif previous is None:
                if drug not in drugs:
                    findings.append(_f(patient, "sem_referencia", INFO,
                                       f"{drug}: medicamento sem referência cadastrada", ts,
                                       tags, {"drug": drug}))
                same = [d for d in active if d != drug and drugs.get(d, {}).get("class") == cls]
                if same and cls in sensitive:
                    findings.append(_f(
                        patient, "duplicidade_classe", ATENCAO,
                        f"{drug} iniciado com {', '.join(same)} ativo(s): duplicidade de "
                        f"classe ({cls})", ts, tags + ["duplicidade"],
                        {"drug": drug, "active_same_class": same}))
                for inter in interactions:
                    pair = set(inter["pair"])
                    if drug in pair:
                        other = next(iter(pair - {drug}))
                        if other in active:
                            findings.append(_f(
                                patient, "interacao", int(inter.get("severity", ATENCAO)),
                                f"Interação {drug} + {other}: {inter['risk']}", ts,
                                tags + ["interacao"], {"pair": sorted(pair)}))
            active[drug] = daily

        # Ignora as primeiras 12 h (prescrição de admissão) e conta alterações em 24 h.
        later = g[g.timestamp > g.timestamp.min() + pd.Timedelta(hours=12)]
        times = later["timestamp"].sort_values()
        counts = pd.Series(1.0, index=pd.DatetimeIndex(times)).rolling("24h").sum()
        busy = counts[counts >= 3]
        if not busy.empty:
            ts = busy.index[0]
            findings.append(_f(
                patient, "instabilidade_terapeutica", INFO,
                f"{int(busy.max())} alterações de prescrição em 24 h (instabilidade terapêutica)",
                ts, ["padrao"], {"max_changes_24h": int(busy.max())}))
    return findings
