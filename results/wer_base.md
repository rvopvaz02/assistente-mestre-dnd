## WER: Whisper `base` (cuda), 10 amostras

| Estrato | n | base | prompt | base+dic_loo | prompt+dic_loo | base+dic_vaz | prompt+llm |
|---|---|---|---|---|---|---|---|
| **Geral** | 10 | 33.6% | 28.3% | 33.6% | 28.3% | 27.6% | 19.1% |
| ambiente=silencio | 10 | 33.6% | 28.3% | 33.6% | 28.3% | 27.6% | 19.1% |
| locutor=eu | 10 | 33.6% | 28.3% | 33.6% | 28.3% | 27.6% | 19.1% |
| **Acerto de termos de D&D** ↑ | 25 termos | 64.0% | 84.0% | 64.0% | 84.0% | 84.0% | 100.0% |

`dic_loo` = dicionário avaliado com leave-one-out; `dic_vaz` = com vazamento (só referência).
