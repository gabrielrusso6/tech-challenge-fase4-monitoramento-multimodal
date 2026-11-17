"""Estruturas compartilhadas entre os módulos."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

INFO, ATENCAO, CRITICO = 1, 2, 3
SEVERITY_LABEL = {INFO: "INFO", ATENCAO: "ATENÇÃO", CRITICO: "CRÍTICO"}
LABEL_SEVERITY = {v: k for k, v in SEVERITY_LABEL.items()}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Finding:
    """Um achado de um módulo (vídeo, áudio, sinais vitais, prescrição, movimentação).

    `timestamp` é o momento da análise; `offset` localiza o achado dentro da mídia ou série
    (ex.: "00:42-00:48" no vídeo, "t=310s" no monitor, "2026-11-12 08:00" na prescrição).
    """

    patient_id: str
    source: str
    kind: str
    severity: int
    message: str
    offset: str = ""
    score: float = 0.0
    tags: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=now_iso)

    @property
    def level(self) -> str:
        return SEVERITY_LABEL[self.severity]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["level"] = self.level
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Finding":
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in keys})


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
