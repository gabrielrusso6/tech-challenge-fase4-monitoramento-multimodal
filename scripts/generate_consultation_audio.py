"""Gera dois áudios SINTÉTICOS de consulta em pt-BR com Azure Text to Speech.

- consulta_estavel.wav: paciente fala em ritmo normal, sem queixas relevantes;
- consulta_alerta.wav: fala lenta, pausas longas e curtas (simula fadiga/dispneia) e queixas
  de falta de ar, cansaço e braço inchado.

Os marcadores vocais são SIMULADOS via SSML (ritmo, pausas, volume); não são voz patológica
real. Uso: python scripts/generate_consultation_audio.py  (requer AZURE_SPEECH_KEY/REGION)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tc4.audio.azure_services import synthesize_ssml  # noqa: E402
from tc4.config import AzureSettings  # noqa: E402

DOCTOR = "pt-BR-AntonioNeural"
PATIENT = "pt-BR-FranciscaNeural"


def ssml(turns: list[tuple[str, str, str]]) -> str:
    body = []
    for voice, prosody, text in turns:
        body.append(f'<voice name="{voice}"><prosody {prosody}>{text}</prosody></voice>')
    return ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            'xml:lang="pt-BR">' + "".join(body) + "</speak>")


NORMAL = 'rate="0%"'
TIRED = 'rate="-30%" pitch="-10%" volume="-25%"'

STABLE = [
    (DOCTOR, NORMAL, "Bom dia, Dona Maria. Como a senhora passou a noite?"),
    (PATIENT, NORMAL, "Bom dia, doutor. Passei bem, dormi a noite toda. A dor no local da "
                      "cirurgia melhorou bastante e hoje consegui caminhar pelo corredor."),
    (DOCTOR, NORMAL, "Ótimo. E os exercícios de ombro com a fisioterapia?"),
    (PATIENT, NORMAL, "Estou fazendo direitinho. O braço está levantando mais a cada dia e "
                      "estou me sentindo animada para ir para casa."),
]

ALERT = [
    (DOCTOR, NORMAL, "Bom dia, Dona Maria. Como a senhora passou a noite?"),
    (PATIENT, TIRED, 'Ai, doutor... <break time="1200ms"/> estou muito cansada. '
                     '<break time="500ms"/> Sinto falta de ar <break time="400ms"/> quando '
                     'tento levantar <break time="500ms"/> e até para falar. '
                     '<break time="1500ms"/>'),
    (DOCTOR, NORMAL, "A senhora sentiu mais alguma coisa?"),
    (PATIENT, TIRED, 'Um aperto no peito <break time="600ms"/> de madrugada. '
                     '<break time="1300ms"/> E uma fraqueza, <break time="400ms"/> sem força '
                     '<break time="500ms"/> para os exercícios. <break time="1200ms"/> '
                     'O braço do lado da cirurgia <break time="400ms"/> está inchado '
                     '<break time="500ms"/> e vermelho.'),
    (DOCTOR, NORMAL, "Certo. Vou pedir para a enfermagem verificar os sinais vitais agora."),
]


def main() -> int:
    settings = AzureSettings.from_env()
    if not settings.speech_ready:
        print("Configure AZURE_SPEECH_KEY e AZURE_SPEECH_REGION no .env", file=sys.stderr)
        return 1
    out = ROOT / "data" / "samples"
    for name, turns in (("consulta_estavel.wav", STABLE), ("consulta_alerta.wav", ALERT)):
        path = synthesize_ssml(ssml(turns), out / name, settings)
        print(f"gerado: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
