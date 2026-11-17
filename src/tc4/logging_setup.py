"""Logging em console e arquivo (runtime/logs/tc4.log) para rastreabilidade."""
from __future__ import annotations

import logging

from .config import runtime_dir

_CONFIGURED = False


def get_logger(name: str = "tc4") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        log_dir = runtime_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        root = logging.getLogger("tc4")
        root.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_dir / "tc4.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(file_handler)
        root.addHandler(console)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("tc4") else f"tc4.{name}")
