"""Orquestra: achados -> fusão -> alertas persistidos -> notificação da equipe."""
from __future__ import annotations

from pathlib import Path

from ..config import runtime_dir
from ..fusion.risk import FusionResult, fuse
from ..logging_setup import get_logger
from ..models import SEVERITY_LABEL, Finding
from .notifier import EmailNotifier
from .store import AlertStore

log = get_logger("tc4.alerts")


def default_db() -> Path:
    return runtime_dir() / "tc4.sqlite"


class AlertService:
    def __init__(self, db_path: Path | None = None, notifier: EmailNotifier | None = None):
        self.store = AlertStore(db_path or default_db())
        self.notifier = notifier or EmailNotifier()

    def record(self, findings: list[Finding]) -> int:
        return self.store.add_findings(findings)

    def raise_alert(self, alert: dict) -> tuple[int, bool]:
        if not isinstance(alert["level"], str):
            alert = {**alert, "level": SEVERITY_LABEL[alert["level"]]}
        alert_id, created = self.store.add_alert(alert)
        if created:
            via = self.notifier.notify(alert, alert_id)
            if via:
                self.store.set_notified(alert_id, via)
            log.info("alerta %d [%s] %s — %s", alert_id, alert["level"], alert["patient_id"],
                     alert["title"])
        return alert_id, created

    def realtime(self, finding: Finding) -> tuple[int, bool] | None:
        """Achados críticos do monitor em tempo real geram alerta imediato (sem esperar fusão)."""
        self.record([finding])
        if finding.severity < 3:
            return None
        return self.raise_alert({
            "patient_id": finding.patient_id, "level": finding.severity,
            "title": f"Tempo real — {finding.kind.replace('_', ' ')}",
            "message": f"{finding.message} ({finding.offset})",
            "sources": [finding.source], "evidence": [finding.to_dict()]})

    def fuse_and_alert(self, patient_id: str, hours: float = 24.0) -> tuple[FusionResult, list[int]]:
        findings = self.store.findings(patient_id, hours=hours)
        result = fuse(findings, patient_id)
        ids = []
        for alert in result.alerts:
            alert_id, _ = self.raise_alert(alert)
            ids.append(alert_id)
        log.info("fusão %s: escore=%.1f nível=%s alertas=%d", patient_id, result.score,
                 SEVERITY_LABEL[result.level], len(ids))
        return result, ids
