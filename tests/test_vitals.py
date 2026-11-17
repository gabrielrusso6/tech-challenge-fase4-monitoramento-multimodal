import numpy as np

from tc4.anomaly.vitals import (EVENT_KINDS, OnlineVitalsMonitor, detect_batch,
                                evaluate_flags, inject_anomalies, keep_runs,
                                segments_to_findings, synthetic_vitals)
from tc4.models import CRITICO


def test_keep_runs_respects_persistence():
    mask = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert keep_runs(mask, 3).tolist() == [0, 0, 0, 1, 1, 1, 0, 0]


def test_all_injected_event_kinds_are_detected():
    df = synthetic_vitals(n=900, seed=3)
    injected, labels, events = inject_anomalies(df, EVENT_KINDS, seed=3)
    det = detect_batch(injected)
    metrics = evaluate_flags(det["ensemble"].to_numpy(), labels, events)
    assert metrics["event_recall"] == 1.0
    assert metrics["false_alarms"] <= 3


def test_clean_series_has_few_alarms():
    df = synthetic_vitals(n=480, seed=11)
    det = detect_batch(df)
    assert not det["rule"].any()
    assert det["ensemble"].mean() < 0.10


def test_findings_are_critical_when_clinical_limit_crossed():
    df = synthetic_vitals(n=480, seed=5)
    injected, _, _ = inject_anomalies(df, ["dessaturacao"], seed=5)
    findings = segments_to_findings(injected, detect_batch(injected), "SYN-T")
    assert any(f.severity == CRITICO and "respiratorio" in f.tags for f in findings)


def test_online_monitor_raises_once_for_tachycardia():
    df = synthetic_vitals(n=480, seed=8)
    injected, _, events = inject_anomalies(df, ["taquicardia"], seed=8)
    monitor = OnlineVitalsMonitor("SYN-T", use_iforest=False)
    found = []
    for row in injected.to_dict("records"):
        found.extend(monitor.update(row))
    critical = [f for f in found if f.severity == CRITICO and f.kind.startswith("hr")]
    assert len(critical) == 1
    t = float(critical[0].offset.split("=")[1].rstrip("s"))
    assert events[0]["start"] <= t <= events[0]["end"]
