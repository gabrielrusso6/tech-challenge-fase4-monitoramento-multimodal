import numpy as np
import soundfile as sf

from tc4.audio.acoustic import acoustic_features, vocal_indicators
from tc4.audio.analyzer import build_findings
from tc4.audio.lexicon import find_critical_terms


def test_lexicon_handles_accents_and_word_boundaries():
    hits = find_critical_terms("Estou com FALTA DE AR e muito cansaço; o braço está inchado.")
    terms = {h["term"] for h in hits}
    assert {"falta de ar", "cansaco"} <= terms
    assert find_critical_terms("a febrezinha passou") == []


def test_acoustic_pauses_detected(tmp_path):
    sr = 16000
    tone = lambda s: 0.3 * np.sin(2 * np.pi * 180 * np.arange(int(sr * s)) / sr)
    silence = lambda s: np.zeros(int(sr * s))
    y = np.concatenate([tone(1.0), silence(1.5), tone(1.0), silence(1.5), tone(1.0),
                        silence(0.5), tone(1.0)])
    path = tmp_path / "a.wav"
    sf.write(path, y.astype("float32"), sr)
    feats = acoustic_features(path)
    assert feats["long_pauses"] == 2
    assert 170 < feats["f0_mean_hz"] < 190
    ind = vocal_indicators(feats, words_per_min=80)
    assert any(c["name"] == "ritmo_lento" and c["flagged"] for c in ind["checks"])


def test_findings_from_terms_and_indicators():
    terms = find_critical_terms("tenho falta de ar")
    ind = {"checks": [{"name": "ritmo_lento", "flagged": True, "group": "fadiga"},
                      {"name": "pausas_longas", "flagged": True, "group": "fadiga"}],
           "fatigue_score": 0.5, "respiratory_flag": False}
    found = build_findings("SYN-T", terms, ind, None)
    assert {f.kind for f in found} == {"termo_respiratorio", "fadiga_vocal"}
