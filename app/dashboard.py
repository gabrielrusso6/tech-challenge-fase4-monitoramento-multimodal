"""Painel Streamlit: central de alertas, monitor em tempo real e análises multimodais.

Execute na raiz do projeto:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from tc4.alerts.service import AlertService  # noqa: E402
from tc4.config import DATA_DIR, AzureSettings, SmtpSettings, runtime_dir  # noqa: E402
from tc4.models import SEVERITY_LABEL, Finding  # noqa: E402

st.set_page_config(page_title="TC4 — Monitoramento multimodal", page_icon="🩺", layout="wide")


@st.cache_resource
def get_service() -> AlertService:
    return AlertService()


svc = get_service()
azure = AzureSettings.from_env()
smtp = SmtpSettings.from_env()

st.sidebar.title("🩺 Monitoramento multimodal")
patient = st.sidebar.text_input("Paciente (ID sintético)", "SYN-101")
st.sidebar.markdown(
    f"- Azure Speech: {'✅' if azure.speech_ready else '⚠️ não configurado'}\n"
    f"- Azure Language: {'✅' if azure.language_ready else '⚠️ não configurado'}\n"
    f"- E-mail SMTP: {'✅' if smtp.ready else '⚠️ caixa de saída local'}")
st.sidebar.caption("Protótipo acadêmico com dados públicos/sintéticos. Todo alerta exige "
                   "revisão humana; não é dispositivo médico.")


def save_upload(upload, folder: str) -> Path:
    path = runtime_dir() / "uploads" / folder / upload.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(upload.getbuffer())
    return path


def findings_table(items: list[dict]) -> None:
    if not items:
        st.info("Nenhum achado.")
        return
    df = pd.DataFrame(items)
    df["nível"] = df["severity"].map(SEVERITY_LABEL)
    st.dataframe(df[["nível", "source", "kind", "offset", "message"]], hide_index=True)


tab_alerts, tab_rt, tab_video, tab_audio, tab_other = st.tabs(
    ["🚨 Central de alertas", "📈 Monitor em tempo real", "🎥 Vídeo", "🎙️ Áudio",
     "💊 Prescrições e movimentação"])

# ----------------------------------------------------------------------------- alertas
with tab_alerts:
    st.subheader("Central de alertas da equipe médica")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("Executar fusão multimodal", type="primary"):
            result, ids = svc.fuse_and_alert(patient)
            st.session_state["fusion"] = result.to_dict()
            st.success(f"Fusão concluída: {len(ids)} alerta(s)")
    fusion = st.session_state.get("fusion")
    if fusion and fusion["patient_id"] == patient:
        with col_b:
            c1, c2, c3 = st.columns(3)
            c1.metric("Escore de risco", f"{fusion['score']:.0f}/100")
            c2.metric("Nível", fusion["level"])
            c3.metric("Fontes com achados", len(fusion["sources"]))
            for c in fusion["composite"]:
                if c["level"] == 3:
                    st.error(f"**{c['name']}** — {c['action']}")
                else:
                    st.warning(f"**{c['name']}** — {c['action']}")

    alerts = svc.store.alerts(patient_id=patient or None)
    novos = [a for a in alerts if a["status"] == "novo"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Alertas novos", len(novos))
    m2.metric("Críticos novos", sum(1 for a in novos if a["level"] == "CRÍTICO"))
    m3.metric("Revisados", len(alerts) - len(novos))
    if alerts:
        st.dataframe(pd.DataFrame(alerts)[["id", "created_at", "level", "title", "status",
                                           "notified_via", "reviewer"]], hide_index=True)
    for a in novos[:15]:
        icon = "🔴" if a["level"] == "CRÍTICO" else "🟠" if a["level"] == "ATENÇÃO" else "⚪"
        with st.expander(f"{icon} #{a['id']} [{a['level']}] {a['title']}"):
            st.text(a["message"])
            with st.form(f"review_{a['id']}"):
                reviewer = st.text_input("Revisor (nome/CRM fictício)")
                decision = st.radio("Decisão", ["reconhecido", "descartado"], horizontal=True)
                note = st.text_input("Observação")
                if st.form_submit_button("Registrar revisão"):
                    try:
                        svc.store.review(a["id"], reviewer, decision, note)
                        st.success("Revisão registrada na auditoria")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
    with st.expander("Trilha de auditoria"):
        st.dataframe(pd.DataFrame(svc.store.audit()),
                     hide_index=True)
    auto_refresh = st.checkbox("Atualizar automaticamente a cada 5 s")

# ----------------------------------------------------------------------------- tempo real
with tab_rt:
    from tc4.anomaly.vitals import (OnlineVitalsMonitor, inject_anomalies, kinds_for_record,
                                    synthetic_vitals)

    st.subheader("Monitor de sinais vitais em tempo real (replay acelerado)")
    c1, c2, c3, c4 = st.columns(4)
    source = c1.selectbox("Fonte", ["PhysioNet BIDMC", "Sintético"])
    record = c2.number_input("Registro BIDMC", 1, 53, 1)
    speed = c3.slider("Velocidade (x)", 1, 60, 20)
    inject = c4.checkbox("Injetar anomalias", True)
    st.caption("PA sistólica/diastólica são sintéticas (o BIDMC não possui pressão arterial).")
    if st.button("▶️ Iniciar monitoramento", type="primary"):
        if source == "Sintético":
            df = synthetic_vitals(seed=int(record))
        else:
            from tc4.cli import _load_record

            df = _load_record(int(record))
        if inject:
            df, _, events = inject_anomalies(df, kinds_for_record(int(record) - 1),
                                             seed=int(record))
            st.caption("Eventos injetados: " + ", ".join(
                f"{e['kind']} ({e['start']}–{e['end']}s)" for e in events))
        monitor = OnlineVitalsMonitor(patient)
        chart1, chart2 = st.empty(), st.empty()
        status = st.empty()
        log_box = st.container()
        for i, row in enumerate(df.to_dict("records")):
            for f in monitor.update(row):
                res = svc.realtime(f)
                if f.severity == 3:
                    st.toast(f"🔴 ALERTA CRÍTICO: {f.message}")
                    log_box.error(f"t={row['t']:.0f}s — {f.message}"
                                  + (f" → alerta #{res[0]}" if res else ""))
                else:
                    log_box.warning(f"t={row['t']:.0f}s — {f.message}")
            if i % 2 == 0 or i == len(df) - 1:
                win = df.iloc[max(0, i - 180):i + 1].set_index("t")
                chart1.line_chart(win[["hr", "resp", "spo2"]], height=220)
                chart2.line_chart(win[["sbp", "dbp"]], height=160)
                status.caption(f"t = {row['t']:.0f} s | FC {row['hr']:.0f} | FR "
                               f"{row['resp']:.0f} | SpO2 {row['spo2']:.0f}% | PA "
                               f"{row['sbp']:.0f}/{row['dbp']:.0f}")
            time.sleep(1.0 / speed)
        st.success("Replay concluído. Veja a Central de alertas.")

# ----------------------------------------------------------------------------- vídeo
with tab_video:
    st.subheader("Análise de vídeo (YOLOv8-pose + YOLOv8)")
    samples = sorted(p for p in (DATA_DIR / "samples").glob("*")
                     if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})
    upload = st.file_uploader("Vídeo de fisioterapia", type=["mp4", "mov", "avi", "mkv"])
    sample = st.selectbox("...ou um arquivo de data/samples", ["—"] + [p.name for p in samples])
    max_s = st.number_input("Limitar a N segundos (0 = inteiro)", 0, 900, 60)
    if st.button("Analisar vídeo", type="primary"):
        path = save_upload(upload, "video") if upload else \
            (DATA_DIR / "samples" / sample if sample != "—" else None)
        if path is None:
            st.error("Envie um vídeo ou escolha um exemplo.")
        else:
            from tc4.cli import run_video

            with st.spinner("Processando quadros com YOLOv8... (CPU pode levar alguns minutos)"):
                result = run_video(patient, path, float(max_s), True)
            st.session_state["video_result"] = result
            svc.record([Finding.from_dict(f) for f in result["findings"]])
    result = st.session_state.get("video_result")
    if result:
        c1, c2 = st.columns(2)
        if "annotated" in result["artifacts"]:
            c1.video(result["artifacts"]["annotated"])
        c2.image(result["artifacts"]["plot"])
        c2.json(result["summary"], expanded=False)
        findings_table(result["findings"])
        st.download_button("Baixar relatório (Markdown)",
                           Path(result["artifacts"]["report"]).read_text("utf-8"),
                           file_name="relatorio_video.md")

# ----------------------------------------------------------------------------- áudio
with tab_audio:
    st.subheader("Análise de áudio (Azure Speech to Text + Azure AI Language)")
    upload = st.file_uploader("Áudio da consulta", type=["wav", "mp3", "m4a", "ogg", "mp4"])
    samples = sorted(p for p in (DATA_DIR / "samples").glob("*")
                     if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg"})
    sample = st.selectbox("...ou um áudio de data/samples", ["—"] + [p.name for p in samples])
    manual = None
    if not azure.speech_ready:
        manual = st.text_area("Azure Speech não configurado: cole uma transcrição manual "
                              "(será marcada como manual)")
    if st.button("Analisar áudio", type="primary"):
        path = save_upload(upload, "audio") if upload else \
            (DATA_DIR / "samples" / sample if sample != "—" else None)
        if path is None:
            st.error("Envie um áudio ou escolha um exemplo.")
        else:
            from tc4.cli import run_audio

            tfile = None
            if manual:
                tfile = runtime_dir() / "uploads" / "audio" / "manual_transcript.txt"
                tfile.parent.mkdir(parents=True, exist_ok=True)
                tfile.write_text(manual, "utf-8")
            with st.spinner("Transcrevendo e analisando..."):
                try:
                    result = run_audio(patient, path, tfile)
                    st.session_state["audio_result"] = result
                    svc.record([Finding.from_dict(f) for f in result["findings"]])
                except Exception as exc:
                    st.error(f"Falha na análise: {exc}")
    result = st.session_state.get("audio_result")
    if result:
        st.caption(json.dumps(result["modes"], ensure_ascii=False))
        st.audio(str(runtime_dir() / "results" / patient / "audio" / "audio_16k.wav"))
        if result.get("transcript"):
            st.markdown("**Transcrição**")
            st.write(result["transcript"]["text"])
        lang = result.get("language")
        if lang:
            c1, c2 = st.columns(2)
            c1.metric("Sentimento geral (Azure)", lang.get("overall_sentiment") or "-")
            c2.metric("Negatividade média", lang.get("negative_mean"))
            if lang.get("health_entities"):
                st.markdown("**Entidades clínicas (Text Analytics for Health)**")
                st.dataframe(pd.DataFrame(lang["health_entities"]),
                             hide_index=True)
            st.write("Frases-chave:", ", ".join(lang.get("key_phrases", [])[:25]))
        st.markdown("**Indicadores vocais (heurísticos)**")
        st.dataframe(pd.DataFrame(result["indicators"]["checks"]),
                     hide_index=True)
        findings_table(result["findings"])

# ----------------------------------------------------------------------------- outros
with tab_other:
    from tc4.cli import PRESCRIPTIONS_CSV, run_movement, run_prescriptions

    st.subheader("Evolução das prescrições")
    pres = pd.read_csv(PRESCRIPTIONS_CSV)
    st.dataframe(pres[pres.patient_id == patient], hide_index=True)
    if st.button("Analisar prescrições"):
        items = run_prescriptions(PRESCRIPTIONS_CSV, patient)
        svc.record(items)
        st.session_state["pres"] = [f.to_dict() for f in items]
    if "pres" in st.session_state:
        findings_table(st.session_state["pres"])

    st.subheader("Movimentação durante a internação (sintético)")
    if st.button("Analisar movimentação"):
        items = run_movement(patient, None)
        svc.record(items)
        st.session_state["mov"] = [f.to_dict() for f in items]
    if "mov" in st.session_state:
        st.image(str(runtime_dir() / "results" / patient / "movement" / "movement.png"))
        findings_table(st.session_state["mov"])

# Atualização automática no fim do script, para não interromper as outras abas.
if auto_refresh:
    time.sleep(5)
    st.rerun()
