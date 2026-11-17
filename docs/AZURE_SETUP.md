# Configuração do Azure (Speech + Language)

Custo esperado: **zero** usando o plano gratuito F0. O projeto usa poucos minutos de áudio.

| Serviço | Plano F0 (gratuito) | Uso no projeto |
|---|---|---|
| Speech (Speech to Text) | 5 h de áudio/mês | transcrição das consultas |
| Speech (Text to Speech neural) | 0,5 milhão de caracteres/mês | gerar áudios sintéticos (~1.500 caracteres) |
| Language (Text Analytics) | cota mensal gratuita de registros de texto | sentimento, frases-chave, Text Analytics for Health |

Se o F0 não estiver disponível (só é permitido um recurso F0 por tipo por assinatura), o
plano S0 cobra centavos de dólar pelo volume deste projeto (Speech: cerca de US$ 1 por hora
de áudio transcrito). Em assinatura corporativa, crie um **alerta de orçamento**
(Cost Management > Budgets, ex.: R$ 50) e confirme a política interna.

## 1. Grupo de recursos
Portal Azure > Resource groups > Create > nome `rg-tc4-fase4`, região **East US**
(East US tem todos os recursos usados, incluindo Text Analytics for Health).

## 2. Recurso de Speech
Portal > pesquisar **Speech** (Speech service) > Create:
- Resource group: `rg-tc4-fase4`; Region: East US; Name: `tc4-speech-<seu-nome>`;
- Pricing tier: **Free F0**.
Após criar: **Keys and Endpoint** > copie `KEY 1` e a região (`eastus`).

## 3. Recurso de Language
Portal > pesquisar **Language** (Language service) > Create > "Continue to create your
resource" (sem recursos personalizados):
- mesmo grupo e região; Name: `tc4-language-<seu-nome>`; Pricing tier: **Free F0**;
- aceite o aviso de IA responsável.
Após criar: **Keys and Endpoint** > copie `KEY 1` e o `Endpoint`.

## 4. Arquivo .env
```bash
cp .env.example .env      # Windows: copy .env.example .env
```
Preencha `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION=eastus`, `AZURE_LANGUAGE_KEY` e
`AZURE_LANGUAGE_ENDPOINT`. O `.env` está no `.gitignore`: nunca faça commit das chaves e
não mostre as chaves no vídeo.

## 5. Validação
```bash
tc4 azure-check
# Speech: OK (HTTP 200)
# Language: OK (sentimento de teste: neutral)
```

## Idioma
Consultas em português: `AZURE_SPEECH_LOCALE=pt-BR`, `AZURE_TEXT_LANGUAGE=pt-BR`,
`AZURE_HEALTH_LANGUAGE=pt`. Para áudio público em inglês use `en-US`, `en`, `en`.
Se o Text Analytics for Health recusar o idioma, o erro fica registrado em
`language.errors` e as demais análises continuam.

## E-mail (opcional)
Gmail: ative verificação em duas etapas e gere uma **senha de app**; use
`SMTP_HOST=smtp.gmail.com`, `SMTP_PORT=587`. Em Microsoft 365 o SMTP AUTH costuma estar
desativado pelo administrador. Sem SMTP, cada alerta vira um arquivo `.eml` em
`runtime/outbox/`, que pode ser aberto no Outlook para a demonstração.
