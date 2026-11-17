from tc4.anomaly.prescriptions import analyze_prescriptions, load_prescriptions
from tc4.cli import PRESCRIPTIONS_CSV
from tc4.models import CRITICO


def _kinds(patient):
    df = load_prescriptions(PRESCRIPTIONS_CSV)
    return [f for f in analyze_prescriptions(df) if f.patient_id == patient]


def test_demo_patient_anomalies():
    found = _kinds("SYN-101")
    kinds = {f.kind for f in found}
    assert {"dose_acima_maximo", "alteracao_abrupta", "duplicidade_classe", "interacao",
            "suspensao_sem_substituto"} <= kinds
    overdose = [f for f in found if f.kind == "dose_acima_maximo"]
    assert overdose[0].severity == CRITICO and "dipirona" in overdose[0].message
    assert any("opioide" in f.tags for f in found if f.kind == "alteracao_abrupta")


def test_stable_patient_has_no_warnings():
    assert [f for f in _kinds("SYN-102") if f.severity >= 2] == []
