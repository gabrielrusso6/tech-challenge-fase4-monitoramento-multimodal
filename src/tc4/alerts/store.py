"""Persistência de achados, alertas e trilha de auditoria em SQLite."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..models import LABEL_SEVERITY, SEVERITY_LABEL, Finding, now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL, patient_id TEXT NOT NULL, source TEXT NOT NULL,
    kind TEXT NOT NULL, severity INTEGER NOT NULL, message TEXT NOT NULL,
    media_offset TEXT, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL, patient_id TEXT NOT NULL, level TEXT NOT NULL,
    title TEXT NOT NULL, message TEXT NOT NULL, sources TEXT NOT NULL,
    evidence TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'novo',
    notified_via TEXT, reviewer TEXT, reviewed_at TEXT, review_note TEXT
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
    alert_id INTEGER, details TEXT
);
"""
DECISIONS = {"reconhecido", "descartado"}


class AlertStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as conn, conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ achados
    def add_findings(self, findings: list[Finding]) -> int:
        with closing(self._conn()) as conn, conn:
            conn.executemany(
                "INSERT INTO findings (created_at, patient_id, source, kind, severity, message,"
                " media_offset, payload) VALUES (?,?,?,?,?,?,?,?)",
                [(f.timestamp, f.patient_id, f.source, f.kind, f.severity, f.message, f.offset,
                  json.dumps(f.to_dict(), ensure_ascii=False)) for f in findings])
        return len(findings)

    def findings(self, patient_id: str, hours: float = 24.0) -> list[Finding]:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT payload FROM findings WHERE patient_id = ? AND created_at >= ? "
                "ORDER BY id", (patient_id, since)).fetchall()
        return [Finding.from_dict(json.loads(r["payload"])) for r in rows]

    def patients(self) -> list[str]:
        with closing(self._conn()) as conn:
            rows = conn.execute("SELECT DISTINCT patient_id FROM findings UNION "
                                "SELECT DISTINCT patient_id FROM alerts").fetchall()
        return sorted(r[0] for r in rows)

    # ------------------------------------------------------------------ alertas
    def add_alert(self, alert: dict, dedup_minutes: int = 30) -> tuple[int, bool]:
        """Insere alerta. Retorna (id, criado). Mesmo paciente+título recente não duplica."""
        level = alert["level"] if isinstance(alert["level"], str) else \
            SEVERITY_LABEL[alert["level"]]
        since = (datetime.now(timezone.utc) - timedelta(minutes=dedup_minutes)).isoformat()
        with closing(self._conn()) as conn, conn:
            row = conn.execute(
                "SELECT id FROM alerts WHERE patient_id = ? AND title = ? AND created_at >= ? "
                "AND status = 'novo'", (alert["patient_id"], alert["title"], since)).fetchone()
            if row:
                return int(row["id"]), False
            cur = conn.execute(
                "INSERT INTO alerts (created_at, patient_id, level, title, message, sources,"
                " evidence) VALUES (?,?,?,?,?,?,?)",
                (now_iso(), alert["patient_id"], level, alert["title"], alert["message"],
                 json.dumps(alert.get("sources", []), ensure_ascii=False),
                 json.dumps(alert.get("evidence", []), ensure_ascii=False)))
            alert_id = int(cur.lastrowid)
            conn.execute("INSERT INTO audit (ts, actor, action, alert_id, details) "
                         "VALUES (?,?,?,?,?)",
                         (now_iso(), "sistema", "alerta_criado", alert_id, level))
        return alert_id, True

    def set_notified(self, alert_id: int, via: str) -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute("UPDATE alerts SET notified_via = ? WHERE id = ?", (via, alert_id))
            conn.execute("INSERT INTO audit (ts, actor, action, alert_id, details) "
                         "VALUES (?,?,?,?,?)", (now_iso(), "sistema", "notificado", alert_id, via))

    def alerts(self, status: str | None = None, patient_id: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM alerts WHERE 1=1", []
        if status:
            sql += " AND status = ?"
            args.append(status)
        if patient_id:
            sql += " AND patient_id = ?"
            args.append(patient_id)
        sql += " ORDER BY id DESC"
        with closing(self._conn()) as conn:
            rows = conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["sources"] = json.loads(d["sources"])
            d["evidence"] = json.loads(d["evidence"])
            d["severity"] = LABEL_SEVERITY.get(d["level"], 1)
            out.append(d)
        return out

    def review(self, alert_id: int, reviewer: str, decision: str, note: str = "") -> None:
        """Revisão humana obrigatória: reconhecer ou descartar com identificação."""
        if decision not in DECISIONS:
            raise ValueError(f"decisão inválida: {decision}")
        if not reviewer.strip():
            raise ValueError("revisor obrigatório")
        with closing(self._conn()) as conn, conn:
            cur = conn.execute(
                "UPDATE alerts SET status = ?, reviewer = ?, reviewed_at = ?, review_note = ? "
                "WHERE id = ? AND status = 'novo'",
                (decision, reviewer.strip(), now_iso(), note, alert_id))
            if cur.rowcount != 1:
                raise ValueError(f"alerta {alert_id} inexistente ou já revisado")
            conn.execute("INSERT INTO audit (ts, actor, action, alert_id, details) "
                         "VALUES (?,?,?,?,?)", (now_iso(), reviewer.strip(), decision,
                                                alert_id, note))

    def audit(self, limit: int = 200) -> list[dict]:
        with closing(self._conn()) as conn:
            rows = conn.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?",
                                (limit,)).fetchall()
        return [dict(r) for r in rows]
