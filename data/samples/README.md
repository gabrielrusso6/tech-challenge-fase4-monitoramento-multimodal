# Mídias de demonstração (não versionadas)

Coloque aqui os arquivos usados no vídeo. Eles NÃO vão para o Git (licença e tamanho).

## Vídeo (fisioterapia)
Fontes com licença livre: Pexels (pexels.com/videos) ou Mixkit. Buscas sugeridas:
"shoulder exercise", "physiotherapy arm", "physical therapy", "arm raise exercise",
"person falling". Prefira: corpo inteiro visível, câmera parada, 30–90 s, uma pessoa.
Registre o link e o autor de cada vídeo no relatório (atribuição).

## Áudio (consulta)
1. Recomendado: `python scripts/generate_consultation_audio.py` gera duas consultas
   sintéticas em pt-BR com Azure Text to Speech (estável x alerta).
2. Áudio público complementar: clipes CC0 de tosse/respiração (ex.: Freesound) ou AudioSet.
   Áudios em inglês exigem `AZURE_SPEECH_LOCALE=en-US` e `AZURE_TEXT_LANGUAGE=en`.
