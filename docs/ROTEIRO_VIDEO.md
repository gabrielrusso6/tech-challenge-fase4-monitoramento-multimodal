# Roteiro do vídeo — versão de 7 minutos

Cobre os 4 pontos exigidos: [1] análise de áudio e vídeo; [2] detecção e resposta a anomalias;
[3] integração com Azure; [4] fluxo final do alerta à equipe médica.

## Antes de gravar (sem gravar) — 10 min

1. PowerShell na pasta do projeto, com o `.venv` ativo:
   ```powershell
   Remove-Item -Recurse -Force runtime -ErrorAction SilentlyContinue
   streamlit run app/dashboard.py
   ```
2. No painel (paciente SYN-101), pré-processar para não esperar na gravação.
   **Não clicar em "Executar fusão multimodal" agora.**
   - Aba **Vídeo**: `exercicio.mp4` → Analisar vídeo; depois `exercicio_alterado.mp4` → Analisar vídeo
     (fica na tela o alterado).
   - Aba **Áudio**: `consulta_estavel.wav` → Analisar áudio; depois `consulta_alerta.wav` →
     Analisar áudio (fica na tela o de alerta).
   - Aba **Monitor em tempo real**: registro 1, velocidade 20, "Injetar anomalias" marcado.
     **Não iniciar.**
3. Abas do navegador, nesta ordem: (a) GitHub no README; (b) portal Azure no grupo `rg-tc4`;
   (c) painel `localhost:8501`.
4. Explorador de Arquivos aberto em `runtime\outbox` (pasta vazia por enquanto).
5. Não abrir o `.env` nem a página de chaves do Azure. Ativar "Não incomodar". Fechar Teams/e-mail.
6. Gravar: `Win + Shift + R` (Ferramenta de Captura), tela inteira, microfone ligado.

Se ensaiar, repita os passos 1 e 2 antes da gravação valendo (o ensaio cria alertas).

## Gravação

### Cena 1 — Abertura · 0:00–0:30 · GitHub (README)
Fala:
> "Olá, sou Gabriel Russo, e este é o Tech Challenge da Fase 4. Nas fases anteriores criei o
> diagnóstico de câncer de mama e um assistente médico. Agora o hospital quer monitorar
> continuamente a paciente internada após a mastectomia, usando vídeo, áudio e dados clínicos,
> e avisar a equipe médica quando surgir um sinal de risco. É um protótipo acadêmico, com dados
> públicos e sintéticos."

### Cena 2 — Arquitetura · 0:30–1:10 · rolar o README até o diagrama
Fala:
> "O sistema tem cinco fontes: vídeo de fisioterapia analisado com YOLOv8-pose, áudio de
> consulta com Azure Speech to Text e Azure AI Language, sinais vitais reais do PhysioNet,
> prescrições e movimentação na internação. Cada módulo gera achados com severidade. Uma camada
> de fusão cruza essas fontes e gera alertas, que vão por e-mail e para um painel onde a equipe
> revisa."

### Cena 3 — Azure · 1:10–1:30 · portal Azure, grupo rg-tc4  [3]
Mostrar os dois recursos (Fala e Idioma). Não abrir chaves.
Fala:
> "Estes são os serviços Azure usados: Speech para transcrever e Language para analisar o
> texto. As chaves ficam fora do repositório. No painel, a barra lateral confirma os dois
> conectados."
Ir para o painel e apontar os ✅ na barra lateral.

### Cena 4 — Áudio · 1:30–2:35 · aba Áudio  [1][2][3]
1. Dar play no áudio por ~6 segundos e pausar.
2. Rolar devagar: transcrição → sentimento → tabela de entidades → indicadores vocais → achados.
Fala:
> "Esta é uma consulta em que a paciente relata falta de ar e cansaço. O Azure Speech to Text
> fez a transcrição. O Azure Language mediu o sentimento, e o Text Analytics for Health
> reconheceu 'falta de ar' como dispneia e 'aperto no peito' como opressão torácica. Além do que
> ela diz, o sistema mede como ela fala: 132 palavras por minuto e pausas longas, sinais de
> fadiga. Resultado: dois achados críticos e fadiga vocal. Na consulta estável, que rodei antes,
> nenhum alerta foi gerado."

### Cena 5 — Vídeo · 2:35–3:30 · aba Vídeo  [1][2]
Mostrar o vídeo anotado (dar play uns segundos), o gráfico e a tabela de achados.
Fala:
> "Aqui é a análise de vídeo. Gravei o mesmo exercício duas vezes: uma correta e outra errada de
> propósito. O YOLOv8-pose detecta 17 pontos do corpo e calcula o ângulo dos ombros em cada
> quadro. No vídeo correto, nenhum achado de atenção. Neste, o sistema detectou assimetria de 92
> graus entre os braços, repetições com amplitude reduzida e queda progressiva de 144 para 103
> graus, que sugere fadiga ou dor. Ele também gera este relatório automático."
Clicar em "Baixar relatório (Markdown)" é opcional; não abrir se for perder tempo.

### Cena 6 — Sinais vitais em tempo real · 3:30–4:20 · aba Monitor em tempo real  [2][4]
Clicar **▶️ Iniciar monitoramento** e narrar enquanto roda (~25 s).
Fala:
> "Agora os sinais vitais de um paciente real do PhysioNet, em replay acelerado, com eventos
> injetados. O sistema combina limites clínicos, um z-score em relação à linha de base do
> próprio paciente e Isolation Forest. [quando aparecer o 1º alerta] Frequência respiratória
> acima do limite: alerta crítico na hora. [FC subindo] Taquicardia. [SpO2 caindo] Queda de
> saturação. Nos testes com dez registros, o detector pegou cem por cento dos eventos, em
> média dois segundos e meio depois do início."

### Cena 7 — Prescrições e movimentação · 4:20–4:50 · aba Prescrições e movimentação  [2]
Clicar **Analisar prescrições** e depois **Analisar movimentação**.
Fala:
> "Nas prescrições, o sistema encontrou dipirona acima da dose máxima, tramadol dobrado de um
> dia para o outro, dois opioides ao mesmo tempo e o anticoagulante suspenso sem substituto. Na
> movimentação, quatro horas de imobilidade, agitação noturna e uma possível queda."

### Cena 8 — Alerta à equipe médica · 4:50–6:25 · aba Central de alertas  [4]
1. Mostrar os alertas de tempo real que já chegaram.
2. Clicar **Executar fusão multimodal**. Mostrar escore 100, nível CRÍTICO e as regras compostas.
Fala:
> "Os alertas de tempo real já chegaram aqui. Agora executo a fusão. Sozinho, cada sinal pode ser
> ruído. Mas quando a paciente diz que tem falta de ar e, ao mesmo tempo, a respiração e a
> saturação se alteram, o sistema eleva para descompensação respiratória crítica. O mesmo vale
> para opioide somado a alteração respiratória e para queda com alteração hemodinâmica."
3. Ir ao Explorador em `runtime\outbox` e dar dois cliques no `.eml` da descompensação
   respiratória (normalmente o primeiro depois dos de tempo real, ex.: `alerta_00004.eml`).
Fala:
> "Cada alerta gera um e-mail para a equipe, com nível, evidências e ação recomendada. Aqui ele
> fica como arquivo porque não configurei servidor de e-mail."
4. Voltar ao painel → abrir (›) o alerta de descompensação respiratória → Revisor
   "Dra. Ana – CRM 00000", decisão **reconhecido**, observação "Paciente avaliada à beira-leito,
   O2 iniciado" → **Registrar revisão**. Mostrar "Revisados = 1" e abrir **Trilha de auditoria**.
Fala:
> "A médica reconhece o alerta e registra a conduta. Fica tudo na trilha de auditoria: quem
> decidiu, quando e por quê. Nenhuma ação é automática: todo alerta passa por revisão humana."

### Cena 9 — Encerramento · 6:25–7:00 · GitHub (README)
Fala:
> "O código, os testes automatizados e o relatório técnico estão no repositório. As limitações
> estão documentadas: áudio de voz sintética, pressão arterial e prescrições sintéticas, e
> limiares que ainda precisam de validação clínica. Como próximos passos: separar a voz do médico
> e da paciente, streaming em tempo real e notificação pelo Teams. Obrigado."

## Depois de gravar
1. Conferir duração (≤ 7 min) e se nenhuma chave aparece.
2. YouTube → Criar → Enviar vídeo → **Não listado** → copiar o link.
