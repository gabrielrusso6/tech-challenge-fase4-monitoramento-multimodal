"""Integração com Azure AI Speech (Speech to Text / TTS) e Azure AI Language (Text Analytics).

Credenciais vêm de variáveis de ambiente (ver .env.example). As chaves nunca são registradas
em log. Falhas do serviço levantam AzureServiceError: não existe "resposta inventada".
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from ..config import AzureSettings
from ..logging_setup import get_logger

log = get_logger("tc4.azure")


class AzureServiceError(RuntimeError):
    pass


# --------------------------------------------------------------------------- Speech

def transcribe_file(wav_path: Path, settings: AzureSettings, timeout_s: float = 900) -> dict:
    """Transcrição contínua de um WAV 16 kHz mono com timestamps por palavra."""
    import azure.cognitiveservices.speech as speechsdk

    if not settings.speech_ready:
        raise AzureServiceError("AZURE_SPEECH_KEY/AZURE_SPEECH_REGION não configurados")
    config = speechsdk.SpeechConfig(subscription=settings.speech_key,
                                    region=settings.speech_region)
    config.speech_recognition_language = settings.speech_locale
    config.request_word_level_timestamps()
    config.output_format = speechsdk.OutputFormat.Detailed
    audio = speechsdk.audio.AudioConfig(filename=str(wav_path))
    recognizer = speechsdk.SpeechRecognizer(speech_config=config, audio_config=audio)

    segments: list[dict] = []
    errors: list[str] = []
    done = threading.Event()

    def on_recognized(evt) -> None:
        result = evt.result
        if result.reason != speechsdk.ResultReason.RecognizedSpeech or not result.text:
            return
        try:
            best = json.loads(result.json).get("NBest", [{}])[0]
        except (ValueError, IndexError):
            best = {}
        segments.append({
            "text": result.text,
            "offset_s": round(result.offset / 1e7, 2),
            "duration_s": round(result.duration / 1e7, 2),
            "confidence": best.get("Confidence"),
            "words": [{"word": w.get("Word"), "offset_s": round(w.get("Offset", 0) / 1e7, 2),
                       "duration_s": round(w.get("Duration", 0) / 1e7, 2)}
                      for w in best.get("Words", [])],
        })

    def on_canceled(evt) -> None:
        details = getattr(evt, "cancellation_details", None)
        if details is not None and details.reason == speechsdk.CancellationReason.Error:
            errors.append(f"{details.error_details}")
        done.set()

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(lambda _evt: done.set())
    log.info("Azure Speech: transcrevendo %s (%s)", Path(wav_path).name, settings.speech_locale)
    recognizer.start_continuous_recognition()
    finished = done.wait(timeout=timeout_s)
    recognizer.stop_continuous_recognition()
    if errors:
        raise AzureServiceError(f"Azure Speech cancelou a transcrição: {errors[0]}")
    if not finished:
        raise AzureServiceError("tempo limite da transcrição excedido")
    text = " ".join(s["text"] for s in segments)
    log.info("Azure Speech: %d segmentos, %d caracteres", len(segments), len(text))
    return {"engine": "azure-speech", "locale": settings.speech_locale, "text": text,
            "segments": segments}


def synthesize_ssml(ssml: str, out_path: Path, settings: AzureSettings) -> Path:
    """Azure Text to Speech (usado para gerar áudios sintéticos de consulta)."""
    import azure.cognitiveservices.speech as speechsdk

    config = speechsdk.SpeechConfig(subscription=settings.speech_key,
                                    region=settings.speech_region)
    config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio = speechsdk.audio.AudioOutputConfig(filename=str(out_path))
    synth = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=audio)
    result = synth.speak_ssml_async(ssml).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        details = getattr(result, "cancellation_details", None)
        raise AzureServiceError(f"TTS falhou: {getattr(details, 'error_details', result.reason)}")
    return out_path


def check_speech_key(settings: AzureSettings) -> str:
    """Valida a chave pedindo um token ao endpoint STS da região (não consome áudio)."""
    url = f"https://{settings.speech_region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    req = urllib.request.Request(url, data=b"", method="POST",
                                 headers={"Ocp-Apim-Subscription-Key": settings.speech_key})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return f"OK (HTTP {resp.status})"
    except urllib.error.HTTPError as exc:
        raise AzureServiceError(f"Speech respondeu HTTP {exc.code}") from exc


# --------------------------------------------------------------------------- Language

def _client(settings: AzureSettings):
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    if not settings.language_ready:
        raise AzureServiceError("AZURE_LANGUAGE_KEY/AZURE_LANGUAGE_ENDPOINT não configurados")
    return TextAnalyticsClient(endpoint=settings.language_endpoint,
                               credential=AzureKeyCredential(settings.language_key))


def chunk_documents(segments: list[str], max_chars: int = 1000) -> list[str]:
    docs, cur = [], ""
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if cur and len(cur) + len(seg) + 1 > max_chars:
            docs.append(cur)
            cur = seg
        else:
            cur = f"{cur} {seg}".strip()
    if cur:
        docs.append(cur)
    return docs


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def analyze_text(docs: list[str], settings: AzureSettings) -> dict:
    """Sentimento (com opinion mining), frases-chave e entidades de saúde (Text Analytics)."""
    client = _client(settings)
    out: dict = {"engine": "azure-language", "language": settings.text_language,
                 "documents": [], "key_phrases": [], "health_entities": [], "errors": []}
    for batch in _batches(docs, 10):
        for doc, res in zip(batch, client.analyze_sentiment(
                batch, language=settings.text_language, show_opinion_mining=True)):
            if res.is_error:
                out["errors"].append(f"sentiment: {res.error.message}")
                continue
            out["documents"].append({
                "text": doc, "sentiment": res.sentiment,
                "scores": {"positive": res.confidence_scores.positive,
                           "neutral": res.confidence_scores.neutral,
                           "negative": res.confidence_scores.negative},
                "sentences": [{
                    "text": s.text, "sentiment": s.sentiment,
                    "negative": s.confidence_scores.negative,
                    "opinions": [{"target": o.target.text, "sentiment": o.target.sentiment,
                                  "assessments": [a.text for a in o.assessments]}
                                 for o in (s.mined_opinions or [])],
                } for s in res.sentences],
            })
        for res in client.extract_key_phrases(batch, language=settings.text_language):
            if res.is_error:
                out["errors"].append(f"key_phrases: {res.error.message}")
            else:
                out["key_phrases"].extend(res.key_phrases)
    try:
        for batch in _batches(docs, 25):
            poller = client.begin_analyze_healthcare_entities(
                batch, language=settings.health_language)
            for res in poller.result():
                if res.is_error:
                    out["errors"].append(f"health: {res.error.message}")
                    continue
                for e in res.entities:
                    assertion = e.assertion
                    out["health_entities"].append({
                        "text": e.text, "category": e.category,
                        "normalized": e.normalized_text,
                        "confidence": round(e.confidence_score, 3),
                        "certainty": getattr(assertion, "certainty", None),
                        "negated": getattr(assertion, "certainty", None) in
                        ("negative", "negativePossible"),
                    })
    except Exception as exc:  # serviço de saúde pode não estar disponível na região/idioma
        out["errors"].append(f"health: {exc}")
        log.warning("Text Analytics for Health indisponível: %s", exc)
    out["key_phrases"] = sorted(set(out["key_phrases"]))
    scores = [d["scores"]["negative"] for d in out["documents"]]
    out["negative_mean"] = round(sum(scores) / len(scores), 3) if scores else None
    labels = [d["sentiment"] for d in out["documents"]]
    out["overall_sentiment"] = max(set(labels), key=labels.count) if labels else None
    log.info("Azure Language: %d docs, %d entidades de saúde, %d erros", len(docs),
             len(out["health_entities"]), len(out["errors"]))
    return out


def check_language_key(settings: AzureSettings) -> str:
    client = _client(settings)
    res = client.analyze_sentiment(["Teste de conexão."], language=settings.text_language)[0]
    if res.is_error:
        raise AzureServiceError(res.error.message)
    return f"OK (sentimento de teste: {res.sentiment})"
