# Assistente de Mestre (D&D) - Projeto Final LLM e IA Generativa

Este repositório contém o desenvolvimento incremental de um assistente generativo para sessões de RPG de mesa (Dungeons & Dragons), orquestrando voz, visão, RAG e agentes.

---

## Release v0.1 - Multimodalidade Baseada em Áudio (Aula 01)

**Objetivo:** Criar um protótipo de voz funcional capaz de transcrever narrações de RPG, avaliando o impacto de jargões técnicos e nomes próprios no pipeline de fala (ASR), e medindo a melhoria ao aplicar um vocabulário de domínio.

**Metodologia:**
1. **Corpus de Áudio:** Gravação de 10 amostras próprias contendo termos específicos de D&D (ex: Beholder, Tiefling, D20, CA).
2. **Tecnologia (ASR):** Utilização do modelo OpenAI Whisper (base) operando em português.
3. **Avaliação:** Cálculo da Taxa de Erro de Palavra (WER - Word Error Rate) utilizando a biblioteca `jiwer`.

**O Problema (Onde o áudio quebra):**
O modelo padrão apresentou extrema dificuldade com o domínio de fantasia. Erros notáveis incluíram:
* "CA do Lich" transcrito como "seado lixo"
* "Mísseis Mágicos" transcrito como "meça e esmásicos"
* "Beholder" transcrito como "be-roder"
* "Clérigo rola um D20" transcrito como "cléago Roland e 20"

**A Solução e Resultados:**
Foi implementado um vocabulário de domínio customizado (dicionário de substituição pós-transcrição) para mapear os erros fonéticos aos termos corretos do jogo.

**Métricas Comprovadas (Evidência):**
* **WER Antes do vocabulário:** 33.11%
* **WER Depois do vocabulário:** 3.31%
* **Melhoria Absoluta:** 29.80%

O protótipo de voz cumpriu o objetivo da v0.1, reduzindo a taxa de erro a níveis aceitáveis para a sequência do pipeline.
