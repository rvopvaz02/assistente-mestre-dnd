## WER: Whisper `large-v3-turbo` (cuda), 10 amostras

| Estrato | n | base | prompt | base+dic_loo | prompt+dic_loo | base+dic_vaz | prompt+llm |
|---|---|---|---|---|---|---|---|
| **Geral** | 10 | 7.9% | 7.2% | 7.9% | 7.2% | 7.9% | 6.6% |
| ambiente=silencio | 10 | 7.9% | 7.2% | 7.9% | 7.2% | 7.9% | 6.6% |
| locutor=eu | 10 | 7.9% | 7.2% | 7.9% | 7.2% | 7.9% | 6.6% |
| **Acerto de termos de D&D** ↑ | 25 termos | 92.0% | 96.0% | 92.0% | 96.0% | 92.0% | 100.0% |

`dic_loo` = dicionário avaliado com leave-one-out; `dic_vaz` = com vazamento (só referência).
