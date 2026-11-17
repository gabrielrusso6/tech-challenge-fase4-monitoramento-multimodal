# Roteiro do vídeo (até 15 min)

Preparação (antes de gravar):
- `.env` preenchido e `tc4 azure-check` OK. **Não mostrar o `.env` na tela.**
- `tc4 download-physionet --records 1-10` e `tc4 vitals-eval --records 1-10` já executados.
- Áudios gerados (`python scripts/generate_consultation_audio.py`) e vídeo público em
  `data/samples/`.
- Banco limpo: apagar `runtime/tc4.sqlite` antes de gravar.
- Painel aberto: `streamlit run app/dashboard.py`. E-mail de destino aberto em outra aba
  (ou a pasta `runtime/outbox/`).

| Tempo | Cena | O que mostrar / falar |
|---|---|---|
| 0:00–1:00 | Abertura | Desafio da fase 4; continuidade fases 1–3; paciente SYN-101 pós-mastectomia; aviso de protótipo |
| 1:00–2:30 | Arquitetura | Diagrama do README: 5 fontes → achados → fusão → alertas → e-mail/painel; serviços Azure usados |
| 2:30–5:00 | Vídeo | Aba Vídeo: analisar o vídeo de fisioterapia; mostrar esqueleto YOLOv8-pose, ângulos, faixa de alerta, tabela de eventos e relatório automático. Explicar YOLOv8-pose x OpenPose |
| 5:00–7:30 | Áudio + Azure | Portal Azure (recursos Speech e Language, sem mostrar chaves). Aba Áudio: `consulta_alerta.wav` → transcrição Azure, sentimento, entidades Text Analytics for Health, termos críticos, indicadores vocais. Comparar rapidamente com `consulta_estavel.wav` |
| 7:30–9:30 | Sinais vitais em tempo real | Aba Monitor: registro BIDMC 1, anomalias injetadas, velocidade 20×; mostrar alerta crítico surgindo (toast) |
| 9:30–10:30 | Avaliação | `reports/vitals_evaluation_summary.csv` e figura: precisão/recall/falsos alarmes por método |
| 10:30–11:30 | Prescrições e movimentação | Aba correspondente: dipirona acima do máximo, opioide duplicado, suspensão de anticoagulante; gráfico de movimentação com imobilidade, agitação e queda |
| 11:30–13:30 | Fusão e alerta à equipe | Central de alertas → "Executar fusão": escore, regras compostas (ex.: descompensação respiratória). Abrir o e-mail recebido. Revisar um alerta (reconhecer) e mostrar a trilha de auditoria |
| 13:30–14:30 | Código e testes | Estrutura `src/tc4/`, `python -m pytest -v` passando, CI no GitHub |
| 14:30–15:00 | Encerramento | Limitações (dados sintéticos, heurísticas, sem validação clínica) e próximos passos |

Dicas: grave em 1080p, aumente a fonte do terminal, limite a análise de vídeo a 60 s
(`--max-seconds 60`) para não esperar na gravação, ou rode antes e mostre o resultado.
