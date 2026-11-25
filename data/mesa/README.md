# Fotos da mesa (avaliação da leitura de dados)

Coloque aqui fotos da mesa (`.jpg` ou `.png`) e anote em [`gabarito.csv`](gabarito.csv) o que cada uma mostra.

## Como fotografar

- Pelo menos 15 fotos, com variação:
  - 1 dado só (d20), vários dados juntos e dados de tipos diferentes (d4, d6, d8, d10, d12, d20).
  - Luz boa e luz ruim. De cima e em ângulo.
  - Dados com números difíceis: 6 e 9, 1 e 7, e o d4 (que é lido de outro jeito).
  - Algumas fotos com miniaturas em um mapa quadriculado.
- O celular serve.

## Gabarito

Uma linha por foto. A coluna `dados` lista `tipo:valor` separados por `;`, na ordem que quiser.
Exemplo: `d20:17;d6:3;d6:5`.
A coluna `condicao` serve para separar os resultados por estrato (por exemplo `luz_boa`, `luz_ruim`, `angulo`).
Fotos sem dados usam `dados` vazio: o assistente não pode inventar dado que não existe.

As fotos ficam fora do Git por padrão (veja `.gitignore`). Só o gabarito e as métricas são versionados.
