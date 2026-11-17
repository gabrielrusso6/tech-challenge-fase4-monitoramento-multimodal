"""Pipeline de áudio: conversão -> acústica local -> Azure Speech -> Azure Language -> achados."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import AzureSettings
from ..logging_setup import get_logger
from ..models import ATENCAO, INFO, Finding
from .acoustic import acoustic_features, to_wav_16k_mono, vocal_indicators
from .lexicon import find_critical_terms

log = get_logger("tc4.audio")

HEALTH_CATEGORIES = {"SymptomOrSign", "Diagnosis", "MedicationName", "Dosage",
                     "BodyStructure", "ExaminationName"}


def _words_per_minute(transcript: dict | None, speech_s: float) -> float | None:
    if not transcript or not transcript.get("text") or speech_s <= 0:
        return None
    words = len(transcript["text"].split())
    return words / (speech_s / 60)


def build_findings(patient_id: str, terms: list[dict], indicators: dict,
                   language: dict | None) -> list[Finding]:
    findings: list[Finding] = []
    for hit in terms:
        findings.append(Finding(
            patient_id=patient_id, source="audio", kind=f"termo_{hit['group']}",
            severity=hit["severity"],
            message=f"Termo crítico na consulta: \"{hit['term']}\" ({hit['group']})",
            tags=[hit["group"]], evidence=hit))
    flagged = [c for c in indicators["checks"] if c["flagged"] and c["group"] == "fadiga"]
    if len(flagged) >= 2:
        findings.append(Finding(
            patient_id=patient_id, source="audio", kind="fadiga_vocal", severity=ATENCAO,
            message="Possível fadiga vocal: " + ", ".join(c["name"] for c in flagged),
            score=indicators["fatigue_score"], tags=["fadiga"],
            evidence={"checks": flagged}))
    if indicators["respiratory_flag"]:
        findings.append(Finding(
            patient_id=patient_id, source="audio", kind="pausas_respiratorias",
            severity=ATENCAO, message="Pausas curtas frequentes na fala (possível dispneia)",
            tags=["respiratorio"],
            evidence={"checks": [c for c in indicators["checks"]
                                 if c["group"] == "respiratorio"]}))
    if language and language.get("negative_mean") is not None and \
            language["negative_mean"] >= 0.6:
        findings.append(Finding(
            patient_id=patient_id, source="audio", kind="sentimento_negativo", severity=INFO,
            message=f"Sentimento predominantemente negativo (Azure; média negativa "
                    f"{language['negative_mean']:.2f})", score=language["negative_mean"],
            tags=["sentimento"], evidence={"overall": language.get("overall_sentiment")}))
    if language:
        symptoms = [e for e in language.get("health_entities", [])
                    if e["category"] == "SymptomOrSign" and not e["negated"]
                    and e.get("normalized")]  # só entidades ligadas a vocabulário clínico (UMLS)
        if symptoms:
            names = sorted({e["normalized"] or e["text"] for e in symptoms})
            findings.append(Finding(
                patient_id=patient_id, source="audio", kind="sintomas_relatados",
                severity=INFO, message="Sintomas identificados pelo Azure Text Analytics for "
                                       "Health: " + ", ".join(names[:12]),
                tags=["sintomas"], evidence={"entities": symptoms[:30]}))
    return findings


def analyze_audio(input_path: Path, patient_id: str, out_dir: Path,
                  settings: AzureSettings | None = None,
                  transcript_file: Path | None = None) -> dict:
    """Analisa um arquivo de áudio (ou vídeo com áudio).

    Se o Azure não estiver configurado, a parte acústica roda localmente e o texto pode vir de
    `transcript_file` (marcado como `manual`). O modo usado fica registrado em `modes`.
    """
    settings = settings or AzureSettings.from_env()
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = to_wav_16k_mono(Path(input_path), out_dir / "audio_16k.wav")
    acoustic = acoustic_features(wav)
    modes = {"speech": "não executado", "language": "não executado"}

    transcript = None
    if settings.speech_ready:
        from .azure_services import transcribe_file

        transcript = transcribe_file(wav, settings)
        modes["speech"] = "Azure Speech to Text"
    elif transcript_file:
        transcript = {"engine": "manual", "text": Path(transcript_file).read_text("utf-8"),
                      "segments": []}
        modes["speech"] = "transcrição manual (Azure não configurado)"
    else:
        log.warning("Azure Speech não configurado: sem transcrição")

    language = None
    text = transcript["text"] if transcript else ""
    if text and settings.language_ready:
        from .azure_services import analyze_text, chunk_documents

        segs = [s["text"] for s in transcript.get("segments", [])] or [text]
        language = analyze_text(chunk_documents(segs), settings)
        modes["language"] = "Azure AI Language (sentimento, frases-chave, Text Analytics for Health)"
    elif text:
        modes["language"] = "somente léxico local (Azure Language não configurado)"

    wpm = _words_per_minute(transcript, acoustic["speech_s"])
    indicators = vocal_indicators(acoustic, wpm)
    terms = find_critical_terms(text)
    findings = build_findings(patient_id, terms, indicators, language)

    result = {"input": Path(input_path).name, "modes": modes, "acoustic": acoustic,
              "words_per_minute": round(wpm, 1) if wpm else None, "indicators": indicators,
              "transcript": transcript, "language": language, "critical_terms": terms,
              "findings": [f.to_dict() for f in findings]}
    (out_dir / "audio_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    log.info("áudio concluído: modo=%s, %d achados", modes, len(findings))
    return result


def render_report(r: dict) -> str:
    from ..models import SEVERITY_LABEL

    lines = [f"# Relatório automático de áudio — {r['input']}", "",
             f"- Transcrição: {r['modes']['speech']}",
             f"- Análise de texto: {r['modes']['language']}", "",
             "## Indicadores vocais (heurísticos)", "",
             "| Indicador | Valor | Alerta | Critério |", "|---|---|---|---|"]
    for c in r["indicators"]["checks"]:
        lines.append(f"| {c['name']} | {c['value']} | {'sim' if c['flagged'] else 'não'} | "
                     f"{c['description']} |")
    if r.get("transcript"):
        lines += ["", "## Transcrição", "", r["transcript"]["text"][:4000]]
    if r.get("language"):
        lang = r["language"]
        lines += ["", "## Azure AI Language", "",
                  f"- Sentimento geral: {lang.get('overall_sentiment')} "
                  f"(negativo médio {lang.get('negative_mean')})",
                  f"- Frases-chave: {', '.join(lang.get('key_phrases', [])[:20])}"]
        ents = [e for e in lang.get("health_entities", [])
                if e["category"] in HEALTH_CATEGORIES]
        if ents:
            lines += ["", "| Entidade | Categoria | Certeza | Confiança |", "|---|---|---|---|"]
            for e in ents[:40]:
                lines.append(f"| {e['text']} | {e['category']} | {e['certainty'] or '-'} | "
                             f"{e['confidence']} |")
        if lang.get("errors"):
            lines += ["", "Avisos do serviço: " + "; ".join(lang["errors"][:5])]
    lines += ["", "## Achados", ""]
    for f in r["findings"]:
        lines.append(f"- **{SEVERITY_LABEL[f['severity']]}** — {f['message']}")
    if not r["findings"]:
        lines.append("Nenhum achado.")
    return "\n".join(lines) + "\n"
