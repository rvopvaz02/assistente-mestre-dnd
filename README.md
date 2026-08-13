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

## Release v0.2 - Memória e Recuperação com Evidência (RAG) (Aula 02)

**Objetivo:** Fornecer conhecimento de domínio para o assistente através da arquitetura RAG (Retrieval-Augmented Generation), implementando ingestão governada, chunking semântico, e validando a política de abstenção.

**Metodologia (Ingestão Governada):**
1. **Corpus:** Extração tática do Capítulo 9 (Regras de Combate) do Livro do Jogador (PDF, 10 páginas).
2. **Chunking por Fronteira Natural:** Utilização do `RecursiveCharacterTextSplitter` para fatiar o texto respeitando parágrafos e frases (`separators=["\n\n", "\n", ".", " "]`), resultando em 57 chunks semânticos. Evitando assim a quebra de regras do RPG ao meio.
3. **Embeddings & Vector Store:** Modelo `paraphrase-multilingual-MiniLM-L12-v2` (otimizado para o nosso idioma) operando localmente no ChromaDB.

**Métricas e Validação (Evidência do Teste):**
* **Sucesso na Recuperação (Recall@k=3):** Ao receber uma pergunta válida ("Quais são os benefícios da meia cobertura?"), o sistema varreu o banco vetorial e retornou o trecho exato (Página 6) que contém a regra.
* **Trava de Abstenção:** Ao receber uma pergunta fora de escopo ("Traços raciais de um Tiefling"), o sistema recuperou fragmentos irrelacionados, forçando o prompt do agente a não alucinar e acionar o protocolo de abstenção ("Eu não sei").
