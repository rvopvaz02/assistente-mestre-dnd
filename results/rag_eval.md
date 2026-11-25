## Recuperação (Recall@k por estrato)

| Configuração | geral R@1 / R@3 / R@5 | rotineira R@1 / R@3 / R@5 | composta R@1 / R@3 / R@5 | adversarial R@1 / R@3 / R@5 |
|---|---|---|---|---|
| bm25 | 61.3% / 82.3% / 85.5% | 68.8% / 81.2% / 81.2% | 30.0% / 75.0% / 85.0% | 100.0% / 100.0% / 100.0% |
| denso | 43.5% / 66.1% / 71.0% | 50.0% / 68.8% / 75.0% | 25.0% / 65.0% / 70.0% | 60.0% / 60.0% / 60.0% |
| hibrido | 54.8% / 77.4% / 82.3% | 62.5% / 81.2% / 87.5% | 40.0% / 70.0% / 75.0% | 60.0% / 80.0% / 80.0% |
| hibrido+rerank | 69.3% / 90.3% / 95.2% | 75.0% / 87.5% / 93.8% | 45.0% / 90.0% / 95.0% | 100.0% / 100.0% / 100.0% |

## Portão de abstenção (score do reranker)

Limiar calibrado no dev: **0.0171**

| Split | n | Abstenção correta | Respondeu quando devia | Acurácia balanceada |
|---|---|---|---|---|
| dev | 22 | 71.4% | 100.0% | 85.7% |
| teste | 22 | 50.0% | 100.0% | 75.0% |

## Ponta a ponta (`gemini-3.5-flash-lite`)



| Estrato | n | Acerto | Abstenção ok | Citou a evidência | Auditoria de citações | p50 (s) | p95 (s) |
|---|---|---|---|---|---|---|---|
| geral | 44 | 95.5% | 95.5% | 93.3% | 98.4% | 1.191 | 7.37 |
| rotineira | 16 | 93.8% | 93.8% | 100.0% | 100.0% | 1.195 | 15.708 |
| composta | 10 | 100.0% | 100.0% | 80.0% | 100.0% | 1.312 | 1.721 |
| negativa | 10 | 90.0% | 90.0% | - | 100.0% | 0.45 | 1.312 |
| adversarial | 8 | 100.0% | 100.0% | 100.0% | 90.0% | 1.168 | 33.494 |
