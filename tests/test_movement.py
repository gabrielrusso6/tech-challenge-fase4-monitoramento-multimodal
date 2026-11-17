from tc4.anomaly.movement import analyze_activity, synthetic_activity


def test_injected_movement_events_detected():
    df, events = synthetic_activity("SYN-T", seed=7)
    kinds = {f.kind for f in analyze_activity(df)}
    assert {"possivel_queda", "imobilidade_prolongada", "agitacao_noturna"} <= kinds


def test_no_events_without_injection():
    df, _ = synthetic_activity("SYN-T", seed=7, inject=False)
    found = analyze_activity(df)
    assert not [f for f in found if f.kind in {"possivel_queda", "imobilidade_prolongada"}]
