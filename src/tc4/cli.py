"""Linha de comando: `tc4 <comando>` (ou `python -m tc4.cli <comando>`)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DATA_DIR, REPORTS_DIR, AzureSettings, runtime_dir
from .logging_setup import get_logger
from .models import SEVERITY_LABEL, Finding

log = get_logger("tc4.cli")
DEFAULT_PATIENT = "SYN-101"
PHYSIONET_DIR = DATA_DIR / "physionet" / "bidmc"
PRESCRIPTIONS_CSV = DATA_DIR / "prescriptions" / "prescriptions_demo.csv"


def _records(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def _out(patient: str, module: str) -> Path:
    path = runtime_dir() / "results" / patient / module
    path.mkdir(parents=True, exist_ok=True)
    return path


def _print_findings(findings: list[Finding]) -> None:
    if not findings:
        print("  (nenhum achado)")
    for f in findings:
        print(f"  [{SEVERITY_LABEL[f.severity]:8}] {f.source:12} {f.offset:22} {f.message}")


def _service():
    from .alerts.service import AlertService

    return AlertService()


# --------------------------------------------------------------------------- comandos

def cmd_download(args) -> int:
    from .anomaly.vitals import download_bidmc

    paths = download_bidmc(_records(args.records), PHYSIONET_DIR)
    print(f"{len(paths)} registros BIDMC em {PHYSIONET_DIR}")
    return 0


def _load_record(n: int):
    from .anomaly.vitals import download_bidmc, load_bidmc_numerics

    path = download_bidmc([n], PHYSIONET_DIR)[0]
    return load_bidmc_numerics(path, seed=n)


def run_vitals(patient: str, record: int | None, csv: Path | None, inject: bool = True
               ) -> list[Finding]:
    from .anomaly.vitals import (detect_batch, inject_anomalies, kinds_for_record,
                                 load_bidmc_numerics, plot_vitals, segments_to_findings,
                                 synthetic_vitals)

    if csv:
        df, name = load_bidmc_numerics(csv), Path(csv).stem
    elif record:
        df, name = _load_record(record), f"bidmc_{record:02d}"
    else:
        df, name = synthetic_vitals(), "sintetico"
    events = []
    if inject:
        df, _, events = inject_anomalies(df, kinds_for_record((record or 1) - 1), seed=record or 0)
    det = detect_batch(df)
    out = _out(patient, "vitals")
    plot_vitals(df, det, events, out / "vitals.png", title=f"{name} — paciente {patient}")
    findings = segments_to_findings(df, det, patient)
    (out / "vitals.json").write_text(json.dumps(
        {"record": name, "injected_events": events,
         "findings": [f.to_dict() for f in findings]}, ensure_ascii=False, indent=2), "utf-8")
    return findings


def cmd_vitals(args) -> int:
    findings = run_vitals(args.patient, args.record, args.csv, inject=not args.no_inject)
    _print_findings(findings)
    _service().record(findings)
    return 0


def cmd_vitals_eval(args) -> int:
    from .anomaly.vitals import (detect_batch, evaluate_records, inject_anomalies,
                                 kinds_for_record, plot_vitals, synthetic_vitals)

    if args.synthetic:
        frames = {f"sintetico_{i}": synthetic_vitals(seed=i) for i in range(1, 11)}
    else:
        frames = {f"bidmc_{n:02d}": _load_record(n) for n in _records(args.records)}
    detail, summary = evaluate_records(frames)
    REPORTS_DIR.mkdir(exist_ok=True)
    detail.to_csv(REPORTS_DIR / "vitals_evaluation_detail.csv", index=False)
    summary.to_csv(REPORTS_DIR / "vitals_evaluation_summary.csv", index=False)
    first = next(iter(frames))
    df, _, events = inject_anomalies(frames[first], kinds_for_record(0), seed=42)
    plot_vitals(df, detect_batch(df), events, REPORTS_DIR / "figures" / "vitals_example.png",
                title=f"{first}: anomalias injetadas (cinza) x detectadas (vermelho)")
    print(summary.to_string(index=False))
    print(f"\nDetalhe por registro: {REPORTS_DIR / 'vitals_evaluation_detail.csv'}")
    return 0


def run_video(patient: str, path: Path, max_seconds: float, annotate: bool) -> dict:
    from .video.analyzer import VideoConfig, analyze_video

    return analyze_video(path, patient, _out(patient, "video"),
                         VideoConfig(max_seconds=max_seconds), annotate=annotate)


def cmd_video(args) -> int:
    result = run_video(args.patient, args.input, args.max_seconds, not args.no_annotate)
    findings = [Finding.from_dict(f) for f in result["findings"]]
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    _print_findings(findings)
    print(f"Relatório: {result['artifacts']['report']}")
    _service().record(findings)
    return 0


def run_audio(patient: str, path: Path, transcript: Path | None) -> dict:
    from .audio.analyzer import analyze_audio

    return analyze_audio(path, patient, _out(patient, "audio"), AzureSettings.from_env(),
                         transcript_file=transcript)


def cmd_audio(args) -> int:
    result = run_audio(args.patient, args.input, args.transcript)
    print("Modos:", json.dumps(result["modes"], ensure_ascii=False))
    if result.get("transcript"):
        print("Transcrição:", result["transcript"]["text"][:600])
    ac = result["acoustic"]
    print(f"Acústica: fala {ac.get('speech_s')} s de {ac.get('duration_s')} s | "
          f"palavras/min {result.get('words_per_minute')}")
    for c in result["indicators"]["checks"]:
        print(f"  {'ALERTA' if c['flagged'] else 'ok    '} {c['name']:<22} {c['value']!s:>8}"
              f"   ({c['description']})")
    findings = [Finding.from_dict(f) for f in result["findings"]]
    _print_findings(findings)
    _service().record(findings)
    return 0


def run_prescriptions(path: Path, patient: str | None = None) -> list[Finding]:
    from .anomaly.prescriptions import analyze_prescriptions, load_prescriptions

    findings = analyze_prescriptions(load_prescriptions(path))
    return [f for f in findings if patient is None or f.patient_id == patient]


def cmd_prescriptions(args) -> int:
    findings = run_prescriptions(args.input, args.patient)
    _print_findings(findings)
    _service().record(findings)
    return 0


def run_movement(patient: str, path: Path | None) -> list[Finding]:
    from .anomaly.movement import (analyze_activity, load_activity, plot_activity,
                                   synthetic_activity)

    if path:
        df = load_activity(path)
    else:
        df, _ = synthetic_activity(patient)
        (DATA_DIR / "synthetic").mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_DIR / "synthetic" / f"movement_{patient}.csv", index=False)
    findings = analyze_activity(df, patient)
    plot_activity(df, findings, _out(patient, "movement") / "movement.png")
    return findings


def cmd_movement(args) -> int:
    findings = run_movement(args.patient, args.input)
    _print_findings(findings)
    _service().record(findings)
    return 0


def cmd_fuse(args) -> int:
    service = _service()
    result, ids = service.fuse_and_alert(args.patient, hours=args.hours)
    data = result.to_dict()
    (_out(args.patient, "fusion") / "fusion.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    print(f"Paciente {args.patient}: escore {data['score']} — nível {data['level']}")
    print("Fontes:", {k: SEVERITY_LABEL[v] for k, v in data["sources"].items()})
    for c in data["composite"]:
        print(f"  Regra composta: {c['name']} -> {SEVERITY_LABEL[c['level']]}")
    print(f"Alertas: {ids}  (e-mail ou runtime/outbox/)")
    return 0


def cmd_demo(args) -> int:
    patient = args.patient
    service = _service()
    print(f"== Demonstração multimodal — paciente {patient} ==")
    steps = []
    print("\n[1] Sinais vitais (PhysioNet BIDMC + anomalias injetadas)")
    f = run_vitals(patient, args.record, None)
    _print_findings(f)
    steps += f
    print("\n[2] Prescrições")
    f = run_prescriptions(PRESCRIPTIONS_CSV, patient)
    _print_findings(f)
    steps += f
    print("\n[3] Movimentação na internação (sintético)")
    f = run_movement(patient, None)
    _print_findings(f)
    steps += f
    if args.video:
        print("\n[4] Vídeo")
        r = run_video(patient, args.video, args.max_seconds, True)
        f = [Finding.from_dict(x) for x in r["findings"]]
        _print_findings(f)
        steps += f
    if args.audio:
        print("\n[5] Áudio")
        r = run_audio(patient, args.audio, args.transcript)
        print("  Modos:", r["modes"])
        f = [Finding.from_dict(x) for x in r["findings"]]
        _print_findings(f)
        steps += f
    service.record(steps)
    print("\n[6] Fusão e alertas")
    args.hours = 24.0
    return cmd_fuse(args)


def cmd_alerts(args) -> int:
    service = _service()
    if args.action == "list":
        for a in service.store.alerts(status=args.status, patient_id=args.patient):
            print(f"#{a['id']:4} {a['created_at']} [{a['level']:8}] {a['patient_id']} "
                  f"{a['title']} | {a['status']} | via={a['notified_via']}")
        return 0
    if args.id is None or not args.reviewer or args.decision is None:
        print("review exige --id, --reviewer e --decision", file=sys.stderr)
        return 2
    service.store.review(args.id, args.reviewer, args.decision, args.note or "")
    print(f"Alerta {args.id}: {args.decision} por {args.reviewer}")
    return 0


def cmd_azure_check(_args) -> int:
    from .audio.azure_services import check_language_key, check_speech_key

    s = AzureSettings.from_env()
    ok = True
    for name, ready, fn in (("Speech", s.speech_ready, check_speech_key),
                            ("Language", s.language_ready, check_language_key)):
        if not ready:
            print(f"{name}: NÃO configurado (veja .env.example)")
            ok = False
            continue
        try:
            print(f"{name}: {fn(s)}")
        except Exception as exc:
            print(f"{name}: ERRO — {exc}")
            ok = False
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tc4", description="Monitoramento multimodal (Fase 4)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download-physionet", help="baixa registros BIDMC (CSV numéricos)")
    s.add_argument("--records", default="1-10")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("vitals", help="detecta anomalias em um registro")
    s.add_argument("--patient", default=DEFAULT_PATIENT)
    s.add_argument("--record", type=int, default=1)
    s.add_argument("--csv", type=Path)
    s.add_argument("--no-inject", action="store_true", help="não injeta anomalias")
    s.set_defaults(func=cmd_vitals)

    s = sub.add_parser("vitals-eval", help="avaliação quantitativa dos detectores")
    s.add_argument("--records", default="1-10")
    s.add_argument("--synthetic", action="store_true", help="usa séries sintéticas (offline)")
    s.set_defaults(func=cmd_vitals_eval)

    s = sub.add_parser("video", help="analisa vídeo clínico")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--patient", default=DEFAULT_PATIENT)
    s.add_argument("--max-seconds", type=float, default=0.0)
    s.add_argument("--no-annotate", action="store_true")
    s.set_defaults(func=cmd_video)

    s = sub.add_parser("audio", help="analisa áudio de consulta (Azure)")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--patient", default=DEFAULT_PATIENT)
    s.add_argument("--transcript", type=Path, help="texto manual se Azure não configurado")
    s.set_defaults(func=cmd_audio)

    s = sub.add_parser("prescriptions", help="analisa evolução das prescrições")
    s.add_argument("--input", type=Path, default=PRESCRIPTIONS_CSV)
    s.add_argument("--patient")
    s.set_defaults(func=cmd_prescriptions)

    s = sub.add_parser("movement", help="analisa movimentação na internação")
    s.add_argument("--patient", default=DEFAULT_PATIENT)
    s.add_argument("--input", type=Path)
    s.set_defaults(func=cmd_movement)

    s = sub.add_parser("fuse", help="fusão multimodal e geração de alertas")
    s.add_argument("--patient", default=DEFAULT_PATIENT)
    s.add_argument("--hours", type=float, default=24.0)
    s.set_defaults(func=cmd_fuse)

    s = sub.add_parser("demo", help="executa o fluxo completo")
    s.add_argument("--patient", default=DEFAULT_PATIENT)
    s.add_argument("--record", type=int, default=1)
    s.add_argument("--video", type=Path)
    s.add_argument("--audio", type=Path)
    s.add_argument("--transcript", type=Path)
    s.add_argument("--max-seconds", type=float, default=0.0)
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("alerts", help="lista ou revisa alertas")
    s.add_argument("action", choices=["list", "review"])
    s.add_argument("--status", choices=["novo", "reconhecido", "descartado"])
    s.add_argument("--patient")
    s.add_argument("--id", type=int)
    s.add_argument("--reviewer")
    s.add_argument("--decision", choices=["reconhecido", "descartado"])
    s.add_argument("--note")
    s.set_defaults(func=cmd_alerts)

    s = sub.add_parser("azure-check", help="valida as chaves do Azure")
    s.set_defaults(func=cmd_azure_check)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
