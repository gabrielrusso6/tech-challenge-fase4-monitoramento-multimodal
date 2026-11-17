"""Notificação da equipe médica por e-mail (SMTP) com fallback para caixa de saída local."""
from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

from ..config import SmtpSettings, runtime_dir
from ..logging_setup import get_logger
from ..models import LABEL_SEVERITY

log = get_logger("tc4.notifier")


def build_email(alert: dict, alert_id: int, sender: str, recipients: tuple[str, ...]
                ) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = f"[{alert['level']}] Paciente {alert['patient_id']} — {alert['title']}"
    msg["From"] = sender or "tc4-monitor@localhost"
    msg["To"] = ", ".join(recipients) or "equipe@localhost"
    msg.set_content(
        f"Alerta #{alert_id} — nível {alert['level']}\n"
        f"Paciente (sintético): {alert['patient_id']}\n"
        f"Fontes: {', '.join(alert.get('sources', []))}\n\n"
        f"{alert['message']}\n\n"
        "Ação necessária: revisar o alerta no painel (reconhecer ou descartar).\n"
        "Protótipo acadêmico — dados sintéticos/públicos; não usar para decisão clínica.\n")
    return msg


class EmailNotifier:
    def __init__(self, settings: SmtpSettings | None = None, min_level: str = "ATENÇÃO",
                 outbox: Path | None = None) -> None:
        self.settings = settings or SmtpSettings.from_env()
        self.min_severity = LABEL_SEVERITY[min_level]
        self.outbox = outbox or runtime_dir() / "outbox"

    def notify(self, alert: dict, alert_id: int) -> str | None:
        level = alert["level"]
        if LABEL_SEVERITY.get(level, 1) < self.min_severity:
            return None
        msg = build_email(alert, alert_id, self.settings.sender, self.settings.recipients)
        if self.settings.ready:
            try:
                with smtplib.SMTP(self.settings.host, self.settings.port, timeout=20) as smtp:
                    if self.settings.starttls:
                        smtp.starttls(context=ssl.create_default_context())
                    if self.settings.user:
                        smtp.login(self.settings.user, self.settings.password)
                    smtp.send_message(msg)
                log.info("alerta %d enviado por e-mail para %d destinatário(s)", alert_id,
                         len(self.settings.recipients))
                return "email"
            except Exception as exc:  # falha de SMTP não pode perder o alerta
                log.error("falha no envio SMTP do alerta %d: %s", alert_id, exc)
        self.outbox.mkdir(parents=True, exist_ok=True)
        path = self.outbox / f"alerta_{alert_id:05d}.eml"
        path.write_bytes(bytes(msg))
        log.info("alerta %d gravado na caixa de saída local: %s", alert_id, path.name)
        return "outbox"
