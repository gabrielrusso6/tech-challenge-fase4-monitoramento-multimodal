import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("TC4_RUNTIME_DIR", str(tmp_path / "runtime"))
    for var in ("SMTP_HOST", "ALERT_EMAIL_TO", "AZURE_SPEECH_KEY", "AZURE_LANGUAGE_KEY"):
        monkeypatch.setenv(var, "")
    yield
