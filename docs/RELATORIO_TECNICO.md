# TECH CHALLENGE — FASE 4
## Monitoramento multimodal de pacientes com IA e serviços Azure

| Campo | Informação |
|---|---|
| Projeto | Monitoramento multimodal (vídeo, áudio, sinais vitais, prescrições e movimentação) com alertas à equipe médica |
| Repositório Git | https://github.com/gabrielrusso6/tech-challenge-fase4-monitoramento-multimodal |
| Vídeo de demonstração | https://youtu.be/87z6JiW3NDU |
| Integrante | Gabriel Augusto Russo |
| Data | 17/11/2026 |

---

## 1. Resumo executivo

Esta fase estende o projeto (fases 1–3, tema câncer de mama) para o **monitoramento contínuo
de uma paciente internada após mastectomia**. O sistema processa cinco fontes:

1. **Vídeo** de sessões de fisioterapia com YOLOv8-pose e YOLOv8 → desvios de execução,
   fadiga, possível queda e relatório automático;
2. **Áudio** de consultas com **Azure Speech to Text** e **Azure AI Language (Text
   Analytics)** → transcrição, sentimento, entidades clínicas, termos críticos e indicadores
   vocais de fadiga/dispneia;
3. **Sinais vitais** reais (PhysioNet BIDMC) → regras clínicas, z-score robusto e
   Isolation Forest, inclusive em tempo real;
4. **Prescrições** → alterações inesperadas no tratamento;
5. **Movimentação** na internação → imobilidade, agitação noturna, queda e declínio.

Os achados são padronizados, fundidos em um escore de risco com regras compostas entre
modalidades e transformados em **alertas** persistidos com auditoria, enviados por e-mail e
revisados por um profissional no painel. Resultado principal: em 10 registros reais do
PhysioNet BIDMC com 20 eventos injetados, o detector combinado de sinais vitais encontrou
**100% dos eventos**, com **F1 0,934**, latência média de **2,5 s** e **2,2 falsos
alarmes/hora**. No áudio, o Azure Speech to Text transcreveu as consultas em pt-BR e o
Azure AI Language identificou dispneia e aperto no peito na consulta de alerta, sem
alertas na consulta estável.

## 2. Fluxo multimodal

```
Vídeo ──► YOLOv8-pose/YOLOv8 ──► eventos biomecânicos ─┐
Áudio ──► Azure STT ──► Azure Text Analytics ──────────┤
      └─► acústica local (pausas, ritmo, F0) ──────────┤
Sinais vitais ──► regras + z-score + Isolation Forest ─┼─► achados ─► fusão ─► alertas ─► e-mail
Prescrições ──► regras de farmacovigilância ───────────┤   (Finding)  (escore,  (SQLite,    + painel
Movimentação ──► perfil circadiano + regras ───────────┘              regras)   auditoria)  (revisão humana)
         └── crítico em tempo real (monitor) ─────────────────────────────────► alerta imediato
```

Todo módulo produz objetos `Finding` (paciente, fonte, tipo, severidade INFO/ATENÇÃO/CRÍTICO,
mensagem, localização na mídia, etiquetas e evidências). A padronização permite fundir dados
heterogêneos e explicar cada alerta pela lista de evidências.

## 3. Dados

| Fonte | Dado | Tratamento |
|---|---|---|
| PhysioNet BIDMC PPG and Respiration (ODC-By 1.0) | 53 registros de 8 min; FC, FR, SpO2 a 1 Hz (pacientes de UTI) | Zeros tratados como ausência, interpolação; PA sistólica/diastólica **sintéticas** correlacionadas à FC; eventos injetados e rotulados (taquicardia, dessaturação, hipotensão, taquipneia, artefato) |
| Prescrições | CSV sintético de 2 pacientes (SYN-101 com eventos, SYN-102 estável) | Tabela de referência educacional de doses, classes e interações |
| Movimentação | Contagens de atividade a cada 10 min, 5 dias, sintéticas | Perfil circadiano, recuperação progressiva e eventos rotulados |
| Vídeo | Dois vídeos gravados pelo próprio autor, com consentimento, no mesmo ambiente e com o celular na vertical: (1) elevação lateral dos dois braços executada corretamente (controle); (2) o mesmo exercício executado errado de propósito (assimetria, amplitude reduzida, fadiga) | Amostragem a ~10 quadros/s (30 fps → 1 a cada 3 quadros; 25 fps → 1 a cada 2) |
| Áudio | Duas consultas sintéticas pt-BR geradas com Azure Text to Speech (vozes Francisca e Antonio; SSML com pausas e ritmo reduzido no caso de alerta) | Conversão para WAV 16 kHz mono |

Nenhum dado real identificável de paciente foi usado.

## 4. Modelos e técnicas por tipo de dado

### 4.1 Vídeo
- **YOLOv8n-pose** (Ultralytics) estima 17 keypoints COCO por pessoa. Foi usado no lugar do
  OpenPose porque executa a mesma tarefa (pose 2D), roda em CPU no macOS e Windows e evita a
  compilação do OpenPose; o enunciado cita OpenPose como exemplo.
- **YOLOv8n** (COCO) detecta objetos do ambiente a cada 5 quadros analisados (cadeira, cama,
  objetos cortantes).
- Features: elevação do ombro (quadril–ombro–cotovelo), cotovelo, joelho, inclinação do
  tronco, assimetria, proporção da caixa, altura do quadril e velocidade.
- Detecção de desvios: picos (repetições) com `scipy.signal.find_peaks`; repetição com
  amplitude < 70% da mediana; assimetria > 25° em exercício bilateral; compensação de tronco
  > 15°; queda progressiva de amplitude (fadiga); possível queda (descida do quadril > 20%
  da altura do quadro em ~1 s e corpo na horizontal); ausência do paciente; segmentos
  atípicos por **Isolation Forest** em janelas de 1 s.
- Saídas: `report.md`, `events.json`, `frames.csv`, `angles.png` e `annotated.mp4` (esqueleto,
  ângulos e faixa de alerta).

### 4.2 Áudio
- **Azure Speech to Text** (reconhecimento contínuo, pt-BR, timestamps por palavra e
  confiança).
- **Azure AI Language (Text Analytics)**: sentimento com opinion mining, frases-chave e
  **Text Analytics for Health** (sintomas, diagnósticos, medicamentos, com asserção de
  negação).
- **Léxico de termos críticos** (pt-BR/inglês) com normalização de acentos, para alertas
  explicáveis ("falta de ar", "dor no peito", "braço inchado"…).
- **Indicadores vocais locais** (NumPy): ritmo (palavras/min), pausas longas e curtas por
  minuto, variação de F0 (monotonia) e queda de energia — heurísticos para fadiga e dispneia.

### 4.3 Sinais vitais
- **Regras clínicas** com persistência de 5 s (limites educacionais: FC 45–120, FR 8–25,
  SpO2 ≥ 92, PAS 90–160, PAD 50–100).
- **Z-score robusto causal** (mediana/MAD dos 120 s anteriores; |z| ≥ 4 por 3 s).
- **Isolation Forest** multivariado (valores + variação em 5 s) treinado nos primeiros 120 s
  de cada paciente (linha de base individual), com persistência de 10 s.
- **Monitor em tempo real** (`OnlineVitalsMonitor`): mesmos detectores amostra a amostra com
  período de silêncio para evitar alertas repetidos.

### 4.4 Prescrições
Regras explicáveis em ordem cronológica: dose diária acima do máximo, alteração ≥ 2× ou
≤ 0,5×, duplicidade de classe sensível (opioide, AINE, anticoagulante), interações da tabela,
suspensão de uso contínuo sem substituto em 24 h e instabilidade (≥ 3 alterações em 24 h).

### 4.5 Movimentação
Imobilidade diurna ≥ 120 min, agitação noturna (z robusto > 3 contra o perfil horário dos
dois primeiros dias), possível queda (pico extremo seguido de 30 min sem movimento) e
declínio funcional (tendência da atividade diária < −15%/dia).

### 4.6 Fusão e alertas
- Escore 0–100 por noisy-OR ponderado por fonte e severidade, +10 quando duas ou mais fontes
  têm achados de atenção.
- Regras compostas: descompensação respiratória (áudio + sinais vitais), depressão
  respiratória associada a opioide (prescrição + sinais vitais), queda com alteração
  hemodinâmica, fadiga/declínio (voz + movimento), risco de sangramento.
- Alertas agregados por fonte para reduzir fadiga de alarme; deduplicação em 30 min.
- Persistência em SQLite com auditoria; notificação por e-mail (SMTP) ou `.eml` local;
  revisão humana obrigatória (reconhecer/descartar com identificação e observação).

## 5. Resultados

### 5.1 Sinais vitais (10 registros BIDMC, 2 eventos injetados por registro)
Registros 1 a 10, 8 min cada, 20 eventos injetados no total (taquicardia, dessaturação,
hipotensão, taquipneia e pico de artefato de 3 s). Saída de `tc4 vitals-eval --records 1-10`.

| Método | Precisão | Recall (pontos) | F1 | Recall de eventos | Latência média (s) | Falsos alarmes/h |
|---|---|---|---|---|---|---|
| Regras clínicas | 0,945 | 0,731 | 0,810 | 0,80 | 10,45 | 1,50 |
| Z-score robusto | 0,983 | 0,872 | 0,923 | 1,00 | 3,10 | 1,50 |
| Isolation Forest | 0,100 | 0,008 | 0,015 | 0,05 | 2,00 | 0,00 |
| **Combinado** | **0,968** | **0,903** | **0,934** | **1,00** | **2,50** | **2,24** |

Figura: `reports/figures/vitals_example.png`. Análise:
- **Regras clínicas** têm alta precisão, mas encontram 80% dos eventos e são as mais lentas
  (10,5 s), porque exigem 5 s acima do limite e o evento precisa cruzar o limite clínico.
  Os 4 eventos não encontrados correspondem em número aos 4 picos de artefato de 3 s, que a
  persistência foi desenhada justamente para ignorar.
- **Z-score robusto** (mediana/MAD causal) é o melhor detector isolado: pega todos os
  eventos em ~3 s, pois detecta desvios relativos à linha de base do próprio paciente.
- **Isolation Forest**, com o limiar conservador adotado (quantil 99,5% da linha de base
  + 0,05, por 10 s), praticamente não disparou. Nesse cenário ele não acrescentou
  informação às regras e ao z-score; um limiar menos rígido aumentaria a sensibilidade ao
  custo de mais falsos alarmes. Fica registrado como resultado negativo.
- **Combinado** une a confirmação clínica das regras com a rapidez do z-score: 100% dos
  eventos, 2,5 s de latência e 2,2 falsos alarmes/hora. Os falsos alarmes vêm da
  variabilidade natural de pacientes de UTI, cuja linha de base já é instável.

### 5.2 Exemplos de anomalias detectadas
| Fonte | Exemplo | Severidade |
|---|---|---|
| Vídeo (controle) | execução correta (27,8 s): 1 pessoa, 8 repetições, amplitude mediana 101° (elevação até a altura dos ombros), braços simétricos (p95 97° × 103°), tendência −1,3°/rep (sem fadiga) → **nenhum achado de atenção** (sem falso alarme); só INFO de entrada/saída do quadro e 2 transições marcadas como atípicas pelo Isolation Forest | — |
| Vídeo (alterado) | vídeo gravado pelo autor (47,8 s, celular vertical, 10 fps analisados): pessoa detectada em 93% dos quadros, 10 repetições, amplitude mediana 137° | — |
| Vídeo (alterado) | repetição 8: diferença de 92° entre ombro esquerdo e direito (assimetria, braço limitado do lado operado); repetição 10: diferença de 30° | ATENÇÃO |
| Vídeo (alterado) | repetições 9 e 10 com amplitude de 91° e 69° contra mediana de 137° (amplitude reduzida) | ATENÇÃO |
| Vídeo (alterado) | amplitude cai de 144° para 103° ao longo da sessão, tendência −6,6°/repetição (possível fadiga ou dor) | ATENÇÃO |
| Vídeo (alterado) | 2 segmentos com padrão de movimento atípico (Isolation Forest); 2 s iniciais sem paciente no quadro | INFO |
| Áudio | "falta de ar" e "aperto no peito"; Azure Health: *Dyspnea*, *Chest tightness* | CRÍTICO |
| Áudio | "cansada", "fraqueza", "sem força"; braço "inchado" e "vermelho" (linfedema/ferida); fadiga vocal (132 palavras/min, 5,2 pausas longas/min) | ATENÇÃO |
| Sinais vitais | registro BIDMC 01: taquicardia injetada (FC 141 bpm, t=196–251 s) e dessaturação injetada (SpO2 87%, t=383–441 s) | CRÍTICO |
| Sinais vitais | registro BIDMC 01: FR 26 irpm em t=3–16 s, antes da janela de injeção (t ≥ 150 s), ou seja, valor do próprio registro real acima do limite de 25 irpm | CRÍTICO |
| Prescrições | dipirona 6.000 mg/dia acima do máximo de referência (4.000 mg/dia) | CRÍTICO |
| Prescrições | tramadol 200 → 400 mg/dia (2,0×); morfina iniciada com tramadol ativo (duplicidade de opioide); cetorolaco + enoxaparina (risco de sangramento); suspensão de enoxaparina sem substituto em 24 h | ATENÇÃO |
| Prescrições | interações tramadol/morfina + ondansetrona; 3 alterações em 24 h | INFO |
| Prescrições (controle) | paciente SYN-102 estável: nenhum achado | — |
| Movimentação | 14/11 07:30 pico de movimento extremo seguido de imobilidade (possível queda) | CRÍTICO |
| Movimentação | 12/11 10:00–13:50 imobilidade diurna por 240 min; 13/11 02:00–03:50 agitação noturna por 120 min | ATENÇÃO |
| Fusão | SYN-101: escore 100/100, nível CRÍTICO, 4 fontes em CRÍTICO (sinais vitais, prescrições, movimentação, áudio) | CRÍTICO |
| Fusão (regras compostas) | descompensação respiratória (áudio + sinais vitais); depressão respiratória associada a opioide (prescrição + sinais vitais); queda com alteração hemodinâmica (movimento + sinais vitais) | CRÍTICO |
| Fusão (regra composta) | fadiga e declínio funcional (voz + movimento) | ATENÇÃO |

**Fluxo final do alerta (`tc4 demo`, execução de 17/11/2026):** a fusão gerou 8 alertas para
SYN-101: 4 das regras compostas e 4 agregados por fonte (dose acima do máximo, termo
cardiovascular no áudio, 2 achados de atenção em prescrições e 2 em áudio). Cada alerta foi
gravado no SQLite com auditoria e notificado por e-mail (sem SMTP configurado, como arquivo
`.eml` em `runtime/outbox/`, de `alerta_00001.eml` a `alerta_00008.eml`). O alerta 1 foi
revisado com `tc4 alerts review --id 1 --reviewer "Dra. Ana" --decision reconhecido`, e o
status passou de *novo* para *reconhecido*, com revisor e observação registrados.

### 5.3 Comparação estável × alerta (áudio)
| Indicador | `consulta_estavel.wav` | `consulta_alerta.wav` |
|---|---|---|
| Duração / fala | 26,4 s / 21,5 s | 46,2 s / 30,6 s |
| Palavras por minuto | 173,1 | 131,6 (limiar < 140) |
| Pausas ≥ 1 s por minuto | 0,0 | 5,2 (limiar > 4) |
| Pausas curtas por minuto | 13,6 | 16,9 (limiar > 18, não disparou) |
| Variação de F0 (Hz) | 68,7 | 74,3 |
| Sentimento (Azure) | sem alerta | geral *mixed*, negatividade média 0,60 |
| Sintomas (Azure Health, ligados a UMLS) | *Pain* (dor em melhora) | *Dyspnea*, *Chest tightness* |
| Achados | 1 INFO | 10: 2 CRÍTICO, 6 ATENÇÃO (inclui fadiga vocal), 2 INFO |

Ajustes feitos após a primeira execução: (1) só entram como sintoma as entidades do Azure
Health vinculadas a vocabulário clínico, pois expressões como "Passei bem" e "animada"
vinham classificadas como *SymptomOrSign*; (2) "inchado" e "vermelho" entraram no léxico
(linfedema e sinal de infecção no pós-mastectomia); (3) o limiar de fala lenta passou de
110 para 140 palavras/min, porque a fala conversacional em pt-BR fica em torno de
150–190. Esses limiares foram calibrados com dois áudios de voz sintética: é uma
calibração demonstrativa, não validada clinicamente. Os indicadores são calculados sobre o
áudio inteiro, incluindo a fala do médico, o que dilui o efeito (ver diarização em §7).

## 6. Segurança, ética e LGPD
- Somente dados públicos/sintéticos; chaves em `.env`, nunca versionadas ou registradas em log.
- Áudio enviado ao Azure apenas na região configurada; em produção, exigir consentimento,
  contrato de tratamento de dados e minimização.
- Nenhuma ação clínica automática: alertas exigem revisão humana registrada.
- Modelos e limiares não validados clinicamente; viés de dados (UTI norte-americana, TTS).

## 7. Limitações e próximos passos
- Calibrar limiares com dados reais anotados; métricas de vídeo/áudio são qualitativas.
- Diarização (Azure Conversation Transcription) para separar médico e paciente.
- Detector de instrumentos cirúrgicos treinado (m2cai16-tool/Cholec80).
- Streaming real (Azure Event Hubs / IoT Hub) e notificação por Teams/Logic Apps.
- Autenticação e controle de acesso para a revisão.

## 8. Reprodutibilidade
Comandos completos no README. Testes automatizados: `python -m pytest -v` (20 testes
aprovados no Windows 11, Python 3.12). CI em GitHub Actions (Ubuntu, macOS, Windows; Python 3.11/3.12).
