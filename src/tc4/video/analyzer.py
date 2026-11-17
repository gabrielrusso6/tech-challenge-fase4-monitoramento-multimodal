"""Análise de vídeo clínico (sessões de fisioterapia) com YOLOv8-pose e YOLOv8.

Fluxo: quadros amostrados -> YOLOv8-pose (17 keypoints, mesmo conceito do OpenPose) ->
ângulos articulares -> repetições e regras biomecânicas + Isolation Forest -> eventos ->
relatório automático (Markdown/JSON), gráfico e vídeo anotado.

YOLOv8 (COCO) detecta pessoas e objetos do ambiente. O COCO não contém instrumentos
cirúrgicos; para vídeos de cirurgia seria necessário treinar um detector próprio (limitação
declarada no relatório).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import torch_device
from ..logging_setup import get_logger
from ..models import ATENCAO, CRITICO, INFO, Finding, fmt_seconds
from ..anomaly.vitals import keep_runs, mask_segments
from .features import FEATURES, add_speed, frame_features, smooth

log = get_logger("tc4.video")
RISK_OBJECTS = {"knife", "scissors"}


@dataclass
class VideoConfig:
    pose_model: str = "yolov8n-pose.pt"
    det_model: str = "yolov8n.pt"
    detect_objects: bool = True
    det_every: int = 5            # roda o detector de objetos a cada N quadros processados
    target_fps: float = 10.0      # quadros analisados por segundo
    conf: float = 0.4
    max_seconds: float = 0.0      # 0 = vídeo inteiro
    rep_prominence: float = 30.0  # graus
    rep_drop_ratio: float = 0.70  # repetição < 70% da amplitude mediana
    asymmetry_deg: float = 25.0
    trunk_comp_deg: float = 15.0
    fall_drop: float = 0.20       # quadril desce 20% da altura do quadro em ~1 s
    fall_aspect: float = 1.0      # caixa mais larga que alta (corpo deitado)
    absent_s: float = 2.0
    iforest_contamination: float = 0.05


# --------------------------------------------------------------------------- eventos

def _event(kind: str, severity: int, start: float, end: float, desc: str, **metrics) -> dict:
    return {"type": kind, "severity": severity, "start_s": round(float(start), 2),
            "end_s": round(float(end), 2), "description": desc,
            "metrics": {k: (round(float(v), 2) if isinstance(v, (int, float, np.floating))
                            else v) for k, v in metrics.items()}}


def detect_video_events(df: pd.DataFrame, fps_eff: float, cfg: VideoConfig | None = None
                        ) -> tuple[list[dict], dict]:
    """Detecta desvios a partir das features por quadro. Retorna (eventos, resumo)."""
    from scipy.signal import find_peaks

    cfg = cfg or VideoConfig()
    events: list[dict] = []
    df = df.reset_index(drop=True)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    t = df["t"].to_numpy(dtype=float)
    summary: dict = {"frames_analyzed": int(len(df)), "fps_analyzed": round(fps_eff, 2),
                     "duration_s": round(float(t[-1] - t[0]), 1) if len(t) else 0.0}
    person = df["person"].to_numpy(dtype=bool) if "person" in df else np.ones(len(df), bool)
    summary["person_coverage"] = round(float(person.mean()), 3) if len(df) else 0.0

    # 1) Repetições do exercício (picos da elevação do ombro)
    sig = df[["shoulder_l", "shoulder_r"]].max(axis=1)
    reps = []
    if sig.notna().sum() >= fps_eff * 3:
        sig_f = sig.interpolate(limit_direction="both").to_numpy()
        peaks, _ = find_peaks(sig_f, prominence=cfg.rep_prominence,
                              distance=max(1, int(fps_eff * 0.8)))
        for p in peaks:
            lo, hi = max(0, p - int(fps_eff)), min(len(df), p + int(fps_eff) + 1)
            win = df.iloc[lo:hi]
            reps.append({"idx": int(p), "t": float(t[p]), "amp": float(sig_f[p]),
                         "amp_l": float(win["shoulder_l"].max()),
                         "amp_r": float(win["shoulder_r"].max()),
                         "trunk": float(win["trunk_incl"].max()),
                         "t0": float(t[lo]), "t1": float(t[hi - 1])})
    summary["repetitions"] = len(reps)
    if len(reps) >= 3:
        amps = np.array([r["amp"] for r in reps])
        median_amp = float(np.median(amps))
        summary["median_amplitude_deg"] = round(median_amp, 1)
        min_side = np.array([np.nanmin([r["amp_l"], r["amp_r"]]) for r in reps])
        bilateral = bool(np.nanmedian(min_side) >= 0.6 * median_amp)
        summary["exercise_bilateral"] = bilateral
        for i, r in enumerate(reps, start=1):
            if r["amp"] < cfg.rep_drop_ratio * median_amp:
                events.append(_event(
                    "amplitude_reduzida", ATENCAO, r["t0"], r["t1"],
                    f"Repetição {i}: amplitude {r['amp']:.0f}° (mediana {median_amp:.0f}°)",
                    repetition=i, amplitude=r["amp"], median=median_amp))
            asym = abs(r["amp_l"] - r["amp_r"])
            if bilateral and not np.isnan(asym) and asym > cfg.asymmetry_deg:
                events.append(_event(
                    "assimetria", ATENCAO, r["t0"], r["t1"],
                    f"Repetição {i}: diferença de {asym:.0f}° entre ombro esquerdo e direito",
                    repetition=i, left=r["amp_l"], right=r["amp_r"]))
            if not np.isnan(r["trunk"]) and r["trunk"] > cfg.trunk_comp_deg:
                events.append(_event(
                    "compensacao_tronco", INFO, r["t0"], r["t1"],
                    f"Repetição {i}: inclinação do tronco de {r['trunk']:.0f}° (compensação)",
                    repetition=i, trunk=r["trunk"]))
        third = max(1, len(amps) // 3)
        first, last = amps[:third].mean(), amps[-third:].mean()
        slope = float(np.polyfit(np.arange(len(amps)), amps, 1)[0])
        summary["amplitude_trend_deg_per_rep"] = round(slope, 2)
        if len(amps) >= 6 and last < 0.85 * first and slope < -1.0:
            events.append(_event(
                "fadiga_progressiva", ATENCAO, reps[-third]["t0"], reps[-1]["t1"],
                f"Amplitude cai ao longo da sessão ({first:.0f}° -> {last:.0f}°): "
                "possível fadiga ou dor", first=first, last=last, slope=slope))
    summary["rom_left_p95"] = round(float(np.nanpercentile(df["shoulder_l"], 95)), 1) \
        if df["shoulder_l"].notna().any() else None
    summary["rom_right_p95"] = round(float(np.nanpercentile(df["shoulder_r"], 95)), 1) \
        if df["shoulder_r"].notna().any() else None

    # 2) Possível queda: quadril desce rapidamente e o corpo fica "deitado"
    hip = df["hip_y"].to_numpy(dtype=float)
    aspect = df["bbox_aspect"].to_numpy(dtype=float)
    w = max(1, int(fps_eff))
    i = w
    while i < len(df):
        prev = hip[i - w:i]
        if not np.isnan(hip[i]) and np.isfinite(prev).any():
            drop = hip[i] - np.nanmin(prev)
            after = aspect[i:i + w]
            if drop > cfg.fall_drop and np.isfinite(after).any() and \
                    np.nanmedian(after) > cfg.fall_aspect:
                events.append(_event(
                    "possivel_queda", CRITICO, t[i - w], t[min(len(t) - 1, i + w)],
                    "Descida brusca do quadril e corpo na horizontal: possível queda",
                    hip_drop=drop, aspect=float(np.nanmedian(after))))
                i += 3 * w
                continue
        i += 1

    # 3) Paciente fora do campo de visão
    for s, e in mask_segments(keep_runs(~person, int(cfg.absent_s * fps_eff)), merge_gap=0):
        events.append(_event("fora_do_quadro", INFO, t[s], t[e - 1],
                             "Paciente não detectado no vídeo", seconds=t[e - 1] - t[s]))

    # 4) Objetos de risco (detector COCO)
    if "objects" in df:
        counts = Counter(o for row in df["objects"].fillna("") for o in row.split(";") if o)
        summary["objects"] = dict(counts.most_common(10))
        risky = df["objects"].fillna("").apply(lambda s: bool(RISK_OBJECTS & set(s.split(";"))))
        for s, e in mask_segments(risky.to_numpy(), merge_gap=int(fps_eff * 5)):
            events.append(_event("objeto_risco", INFO, t[s], t[e - 1],
                                 "Objeto cortante detectado na área do paciente"))
    if "n_persons" in df:
        summary["max_persons"] = int(df["n_persons"].max()) if len(df) else 0

    # 5) Padrão atípico não supervisionado (Isolation Forest por janelas de 1 s)
    events.extend(_iforest_windows(df, fps_eff, cfg))
    events.sort(key=lambda e: e["start_s"])
    summary["events"] = dict(Counter(e["type"] for e in events))
    return events, summary


def _iforest_windows(df: pd.DataFrame, fps_eff: float, cfg: VideoConfig) -> list[dict]:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    cols = ["shoulder_l", "shoulder_r", "trunk_incl", "speed", "asymmetry", "bbox_aspect"]
    w = max(2, int(fps_eff))
    rows, spans = [], []
    for s in range(0, len(df) - w + 1, w):
        win = df.iloc[s:s + w][cols]
        if win.notna().mean().mean() < 0.6:
            continue
        rows.append(np.concatenate([win.mean().to_numpy(), win.std().to_numpy()]))
        spans.append((float(df["t"].iloc[s]), float(df["t"].iloc[s + w - 1])))
    if len(rows) < 20:
        return []
    X = np.nan_to_num(np.asarray(rows, dtype=float))
    X = StandardScaler().fit_transform(X)
    model = IsolationForest(contamination=cfg.iforest_contamination, random_state=0).fit(X)
    flagged = model.predict(X) == -1
    out = []
    for s, e in mask_segments(flagged, merge_gap=1):
        out.append(_event("movimento_atipico", INFO, spans[s][0], spans[e - 1][1],
                          "Segmento com padrão de movimento atípico (Isolation Forest)",
                          windows=e - s))
    return out


def events_to_findings(events: list[dict], patient_id: str) -> list[Finding]:
    tags = {"possivel_queda": ["queda", "mobilidade"],
            "fadiga_progressiva": ["fadiga", "mobilidade"],
            "amplitude_reduzida": ["mobilidade", "dor"],
            "assimetria": ["mobilidade"], "compensacao_tronco": ["mobilidade"]}
    return [Finding(patient_id=patient_id, source="video", kind=e["type"],
                    severity=e["severity"], message=e["description"],
                    offset=f"{fmt_seconds(e['start_s'])}-{fmt_seconds(e['end_s'])}",
                    tags=tags.get(e["type"], []), evidence=e["metrics"])
            for e in events]


# --------------------------------------------------------------------------- extração

def _fmt(v: float) -> str:
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0f}"


def extract_pose_sequence(video_path: Path, cfg: VideoConfig, annotated_path: Path | None
                          ) -> tuple[pd.DataFrame, float]:
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"não foi possível abrir o vídeo: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / cfg.target_fps)))
    fps_eff = fps / step
    device = torch_device()
    log.info("video=%s fps=%.1f step=%d device=%s", Path(video_path).name, fps, step, device)
    pose = YOLO(cfg.pose_model)
    det = YOLO(cfg.det_model) if cfg.detect_objects else None
    writer = None
    rows, kps_seq = [], []
    idx, processed = -1, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        if cfg.max_seconds and idx / fps > cfg.max_seconds:
            break
        h, w = frame.shape[:2]
        res = pose.predict(frame, conf=cfg.conf, verbose=False, device=device)[0]
        row = {"frame": idx, "t": idx / fps, "person": False, "n_persons": 0}
        kp_main = None
        if res.boxes is not None and len(res.boxes) > 0 and res.keypoints is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            j = int(np.argmax(areas))
            kp_main = res.keypoints.data.cpu().numpy()[j]
            row.update(frame_features(kp_main, tuple(boxes[j]), h))
            row["person"] = True
            row["n_persons"] = int(len(boxes))
        annotated = res.plot() if annotated_path is not None else None
        objects = []
        if det is not None and processed % cfg.det_every == 0:
            dres = det.predict(frame, conf=cfg.conf, verbose=False, device=device)[0]
            if dres.boxes is not None:
                for b, c in zip(dres.boxes.xyxy.cpu().numpy(),
                                dres.boxes.cls.cpu().numpy().astype(int)):
                    name = dres.names[int(c)]
                    if name == "person":
                        continue
                    objects.append(name)
                    if annotated is not None:
                        x1, y1, x2, y2 = map(int, b)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(annotated, name, (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        row["objects"] = ";".join(objects)
        rows.append(row)
        kps_seq.append(kp_main)
        if annotated is not None:
            text = (f"t={row['t']:.1f}s  ombro E {_fmt(row.get('shoulder_l'))}  "
                    f"D {_fmt(row.get('shoulder_r'))}  tronco {_fmt(row.get('trunk_incl'))}")
            cv2.rectangle(annotated, (0, h - 34), (w, h), (0, 0, 0), -1)
            cv2.putText(annotated, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            if writer is None:
                annotated_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps_eff, (w, h))
            writer.write(annotated)
        processed += 1
        if processed % 100 == 0:
            log.info("quadros processados: %d", processed)
    cap.release()
    if writer is not None:
        writer.release()
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("nenhum quadro processado; verifique o arquivo de vídeo")
    for col in FEATURES + ["bbox_h"]:
        if col not in df.columns:
            df[col] = np.nan
    df = add_speed(df, kps_seq, fps_eff)
    return df, fps_eff


def _burn_events_and_encode(src: Path, dst: Path, events: list[dict]) -> Path:
    """Escreve faixa vermelha nos trechos com eventos e converte para H.264 (navegador)."""
    import cv2

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    tmp = src.with_name(src.stem + "_ev.mp4")
    writer = None
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = k / fps
        k += 1
        active = [e for e in events if e["start_s"] <= t <= e["end_s"] and e["severity"] >= 2]
        if active:
            ev = max(active, key=lambda e: e["severity"])
            color = (0, 0, 220) if ev["severity"] == CRITICO else (0, 140, 255)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), color, -1)
            cv2.putText(frame, f"ALERTA: {ev['type']}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    cap.release()
    if writer is not None:
        writer.release()
    try:
        import imageio_ffmpeg

        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i",
                        str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags",
                        "+faststart", str(dst)], check=True)
        tmp.unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover - depende do ffmpeg local
        log.warning("conversão H.264 falhou (%s); mantendo mp4v", exc)
        shutil.move(str(tmp), str(dst))
    return dst


def plot_angles(df: pd.DataFrame, events: list[dict], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["t"], df["shoulder_l"], label="ombro esquerdo", lw=1)
    ax.plot(df["t"], df["shoulder_r"], label="ombro direito", lw=1)
    ax.plot(df["t"], df["trunk_incl"], label="inclinação do tronco", lw=1, alpha=0.7)
    colors = {CRITICO: "#c00000", ATENCAO: "#e69500", INFO: "#7f7f7f"}
    for e in events:
        ax.axvspan(e["start_s"], e["end_s"], color=colors[e["severity"]], alpha=0.2)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("graus")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Ângulos articulares e eventos detectados")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def write_report(summary: dict, events: list[dict], video_name: str, path: Path) -> Path:
    from ..models import SEVERITY_LABEL

    lines = [f"# Relatório automático de vídeo — {video_name}", "",
             "Gerado por YOLOv8-pose + regras biomecânicas + Isolation Forest. "
             "Apoio à equipe; não é laudo.", "", "## Resumo", ""]
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Desvios e eventos", ""]
    if not events:
        lines.append("Nenhum desvio detectado com os limiares atuais.")
    else:
        lines += ["| Início | Fim | Severidade | Tipo | Descrição |", "|---|---|---|---|---|"]
        for e in events:
            lines.append(f"| {fmt_seconds(e['start_s'])} | {fmt_seconds(e['end_s'])} | "
                         f"{SEVERITY_LABEL[e['severity']]} | {e['type']} | {e['description']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def analyze_video(video_path: Path, patient_id: str, out_dir: Path,
                  cfg: VideoConfig | None = None, annotate: bool = True) -> dict:
    cfg = cfg or VideoConfig()
    video_path = Path(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_annot = out_dir / "annotated_raw.mp4" if annotate else None
    df, fps_eff = extract_pose_sequence(video_path, cfg, raw_annot)
    sm = smooth(df)
    events, summary = detect_video_events(sm, fps_eff, cfg)
    summary["video"] = video_path.name
    sm.to_csv(out_dir / "frames.csv", index=False)
    (out_dir / "events.json").write_text(json.dumps(
        {"summary": summary, "events": events, "config": asdict(cfg)}, ensure_ascii=False,
        indent=2), encoding="utf-8")
    artifacts = {"frames": str(out_dir / "frames.csv"), "events": str(out_dir / "events.json"),
                 "plot": str(plot_angles(sm, events, out_dir / "angles.png")),
                 "report": str(write_report(summary, events, video_path.name,
                                            out_dir / "report.md"))}
    if raw_annot is not None and raw_annot.exists():
        artifacts["annotated"] = str(_burn_events_and_encode(
            raw_annot, out_dir / "annotated.mp4", events))
        raw_annot.unlink(missing_ok=True)
    findings = events_to_findings(events, patient_id)
    log.info("video concluído: %d eventos, %d repetições", len(events),
             summary.get("repetitions", 0))
    return {"summary": summary, "events": events, "findings": [f.to_dict() for f in findings],
            "artifacts": artifacts}
