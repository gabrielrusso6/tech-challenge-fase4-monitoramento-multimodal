"""Fusão multimodal: combina achados de vídeo, áudio, sinais vitais, prescrição e movimentação.

1. Escore de risco 0–100 por noisy-OR ponderado por fonte e severidade, com bônus quando
   duas ou mais fontes independentes concordam (corroboração).
2. Regras compostas explicáveis (ex.: termo respiratório no áudio + dessaturação = CRÍTICO).
3. Geração de alertas agregados para reduzir fadiga de alarme.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models import ATENCAO, CRITICO, INFO, SEVERITY_LABEL, Finding

SOURCE_WEIGHT = {"vitals": 1.0, "video": 0.8, "audio": 0.7, "prescription": 0.8,
                 "movement": 0.7}
SEVERITY_P = {INFO: 0.10, ATENCAO: 0.35, CRITICO: 0.70}
SOURCE_LABEL = {"vitals": "sinais vitais", "video": "vídeo", "audio": "áudio",
                "prescription": "prescrições", "movement": "movimentação"}

COMPOSITE_RULES = [
    {"name": "Possível descompensação respiratória (áudio + sinais vitais)",
     "needs": [({"audio"}, {"respiratorio"}), ({"vitals"}, {"respiratorio"})],
     "level": CRITICO,
     "action": "Avaliar à beira-leito: SpO2, frequência respiratória e ausculta."},
    {"name": "Possível depressão respiratória associada a opioide",
     "needs": [({"prescription"}, {"opioide"}), ({"vitals"}, {"respiratorio"})],
     "level": CRITICO,
     "action": "Revisar dose de opioide e nível de sedação; considerar protocolo institucional."},
    {"name": "Queda com alteração hemodinâmica",
     "needs": [({"video", "movement"}, {"queda"}), ({"vitals"}, {"hemodinamico"})],
     "level": CRITICO,
     "action": "Avaliar paciente imediatamente; verificar trauma e sinais vitais."},
    {"name": "Fadiga e declínio funcional (voz + movimento)",
     "needs": [({"audio"}, {"fadiga"}), ({"video", "movement"}, {"fadiga", "mobilidade"})],
     "level": ATENCAO,
     "action": "Reavaliar plano de reabilitação, dor, anemia e estado geral."},
    {"name": "Risco de sangramento por interação + sangramento relatado",
     "needs": [({"prescription"}, {"interacao", "anticoagulante"}), ({"audio"}, {"hemorragico"})],
     "level": CRITICO,
     "action": "Revisar anticoagulação/AINE e examinar sítio cirúrgico."},
]


@dataclass
class FusionResult:
    patient_id: str
    score: float
    level: int
    sources: dict[str, int]
    composite: list[dict] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"patient_id": self.patient_id, "score": self.score,
                "level": SEVERITY_LABEL[self.level], "sources": self.sources,
                "composite": self.composite, "alerts": self.alerts}


def _matches(f: Finding, sources: set[str], tags: set[str]) -> bool:
    return f.source in sources and bool(tags & set(f.tags)) and f.severity >= ATENCAO


def fuse(findings: list[Finding], patient_id: str) -> FusionResult:
    fs, seen = [], set()
    for f in findings:  # ignora repetições do mesmo achado (reanálises da mesma mídia)
        key = (f.source, f.kind, f.offset, f.message)
        if f.patient_id == patient_id and key not in seen:
            seen.add(key)
            fs.append(f)
    keep = 1.0
    for f in fs:
        keep *= 1 - SOURCE_WEIGHT.get(f.source, 0.5) * SEVERITY_P[f.severity]
    sources: dict[str, int] = {}
    for f in fs:
        sources[f.source] = max(sources.get(f.source, 0), f.severity)
    corroborating = sum(1 for s in sources.values() if s >= ATENCAO)
    score = 100 * (1 - keep) + (10 if corroborating >= 2 else 0)
    score = round(min(100.0, score), 1)

    composite = []
    covered: set[int] = set()
    for rule in COMPOSITE_RULES:
        groups = [[i for i, f in enumerate(fs) if _matches(f, src, tags)]
                  for src, tags in rule["needs"]]
        if all(groups):
            ids = sorted({i for g in groups for i in g})
            covered.update(ids)
            composite.append({"name": rule["name"], "level": rule["level"],
                              "action": rule["action"], "finding_idx": ids})

    level = INFO
    if any(c["level"] == CRITICO for c in composite) or any(f.severity == CRITICO for f in fs) \
            or score >= 80:
        level = CRITICO
    elif score >= 40 or any(f.severity == ATENCAO for f in fs) or composite:
        level = ATENCAO

    alerts = []
    for c in composite:
        ev = [fs[i] for i in c["finding_idx"]]
        alerts.append({
            "patient_id": patient_id, "level": c["level"], "title": c["name"],
            "message": c["action"] + "\nEvidências:\n" + "\n".join(
                f"- [{SOURCE_LABEL.get(f.source, f.source)}] {f.message} ({f.offset})"
                for f in ev),
            "sources": sorted({f.source for f in ev}),
            "evidence": [f.to_dict() for f in ev]})
    for i, f in enumerate(fs):
        if f.severity == CRITICO and i not in covered:
            alerts.append({
                "patient_id": patient_id, "level": CRITICO,
                "title": (f"{SOURCE_LABEL.get(f.source, f.source).capitalize()}: "
                          + (f.message if len(f.message) <= 90 else f.message[:87] + "...")),
                "message": f"{f.message} ({f.offset})", "sources": [f.source],
                "evidence": [f.to_dict()]})
    by_source: dict[str, list[Finding]] = {}
    for i, f in enumerate(fs):
        if f.severity == ATENCAO and i not in covered:
            by_source.setdefault(f.source, []).append(f)
    for src, items in by_source.items():
        alerts.append({
            "patient_id": patient_id, "level": ATENCAO,
            "title": f"{len(items)} achado(s) de atenção em {SOURCE_LABEL.get(src, src)}",
            "message": "\n".join(f"- {f.message} ({f.offset})" for f in items),
            "sources": [src], "evidence": [f.to_dict() for f in items]})
    return FusionResult(patient_id, score, level, sources, composite, alerts)
