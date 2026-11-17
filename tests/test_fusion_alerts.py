from tc4.alerts.notifier import EmailNotifier
from tc4.alerts.service import AlertService
from tc4.config import SmtpSettings
from tc4.fusion.risk import fuse
from tc4.models import ATENCAO, CRITICO, INFO, Finding


def F(source, sev, tags, kind="k", offset="t"):
    return Finding("SYN-T", source, kind, sev, f"{source} {kind}", offset=offset, tags=tags)


def test_respiratory_corroboration_is_critical():
    res = fuse([F("audio", ATENCAO, ["respiratorio"]), F("vitals", ATENCAO, ["respiratorio"])],
               "SYN-T")
    assert res.level == CRITICO
    assert any("respiratória" in c["name"] for c in res.composite)
    assert res.alerts[0]["level"] == CRITICO


def test_single_info_stays_low():
    res = fuse([F("audio", INFO, ["sentimento"])], "SYN-T")
    assert res.level == INFO and res.score < 20 and res.alerts == []


def test_duplicate_findings_do_not_inflate_score():
    one = fuse([F("video", ATENCAO, ["mobilidade"])], "SYN-T").score
    two = fuse([F("video", ATENCAO, ["mobilidade"])] * 2, "SYN-T").score
    assert one == two


def test_service_dedup_review_and_outbox(tmp_path):
    notifier = EmailNotifier(SmtpSettings(), outbox=tmp_path / "outbox")
    svc = AlertService(tmp_path / "db.sqlite", notifier)
    svc.record([F("audio", CRITICO, ["respiratorio"], "termo"),
                F("vitals", CRITICO, ["respiratorio"], "spo2")])
    result, ids = svc.fuse_and_alert("SYN-T")
    assert ids and result.level == CRITICO
    _, ids2 = svc.fuse_and_alert("SYN-T")
    assert ids == ids2  # sem duplicar alertas
    assert list((tmp_path / "outbox").glob("*.eml"))
    svc.store.review(ids[0], "Dra. Teste", "reconhecido", "avaliado")
    actions = [a["action"] for a in svc.store.audit()]
    assert "reconhecido" in actions and "notificado" in actions
    import pytest
    with pytest.raises(ValueError):
        svc.store.review(ids[0], "Outro", "descartado")
