"""Caminhos, variáveis de ambiente e limiares do projeto.

Nenhuma chave é gravada em código. As credenciais vêm de variáveis de ambiente ou de um
arquivo `.env` na raiz (carregado com python-dotenv, se instalado).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def runtime_dir() -> Path:
    path = Path(os.getenv("TC4_RUNTIME_DIR", str(ROOT / "runtime")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_env() -> None:
    """Carrega `.env` da raiz do projeto quando python-dotenv estiver disponível."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependência opcional
        return
    load_dotenv(ROOT / ".env", override=False)


@dataclass(frozen=True)
class AzureSettings:
    speech_key: str = ""
    speech_region: str = ""
    speech_locale: str = "pt-BR"
    language_key: str = ""
    language_endpoint: str = ""
    text_language: str = "pt-BR"
    health_language: str = "pt"

    @classmethod
    def from_env(cls) -> "AzureSettings":
        load_env()
        return cls(
            speech_key=os.getenv("AZURE_SPEECH_KEY", "").strip(),
            speech_region=os.getenv("AZURE_SPEECH_REGION", "").strip(),
            speech_locale=os.getenv("AZURE_SPEECH_LOCALE", "pt-BR").strip(),
            language_key=os.getenv("AZURE_LANGUAGE_KEY", "").strip(),
            language_endpoint=os.getenv("AZURE_LANGUAGE_ENDPOINT", "").strip(),
            text_language=os.getenv("AZURE_TEXT_LANGUAGE", "pt-BR").strip(),
            health_language=os.getenv("AZURE_HEALTH_LANGUAGE", "pt").strip(),
        )

    @property
    def speech_ready(self) -> bool:
        return bool(self.speech_key and self.speech_region)

    @property
    def language_ready(self) -> bool:
        return bool(self.language_key and self.language_endpoint)


@dataclass(frozen=True)
class SmtpSettings:
    host: str = ""
    port: int = 587
    user: str = ""
    password: str = ""
    sender: str = ""
    recipients: tuple[str, ...] = ()
    starttls: bool = True

    @classmethod
    def from_env(cls) -> "SmtpSettings":
        load_env()
        to = tuple(x.strip() for x in os.getenv("ALERT_EMAIL_TO", "").split(",") if x.strip())
        return cls(
            host=os.getenv("SMTP_HOST", "").strip(),
            port=int(os.getenv("SMTP_PORT", "587") or 587),
            user=os.getenv("SMTP_USER", "").strip(),
            password=os.getenv("SMTP_PASSWORD", ""),
            sender=os.getenv("ALERT_EMAIL_FROM", os.getenv("SMTP_USER", "")).strip(),
            recipients=to,
            starttls=os.getenv("SMTP_STARTTLS", "true").lower() != "false",
        )

    @property
    def ready(self) -> bool:
        return bool(self.host and self.sender and self.recipients)


def torch_device() -> str:
    """Dispositivo para YOLO. Padrão: CUDA se houver, senão CPU. MPS só se pedido."""
    wanted = os.getenv("TC4_DEVICE", "auto").lower()
    if wanted != "auto":
        return wanted
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # pragma: no cover
        pass
    return "cpu"
