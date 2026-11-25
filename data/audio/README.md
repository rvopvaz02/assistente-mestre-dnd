# Corpus de áudio próprio

Coloque aqui os arquivos de áudio listados em [`refs.csv`](refs.csv) (`.wav`, `.mp3`, `.m4a` ou `.ogg`).
A coluna `referencia` é a transcrição exata do que foi falado: leia a frase **como está escrita**.

## Como gravar

- Pelo menos 10 amostras. As frases já estão no `refs.csv`; você pode trocar por outras, desde que atualize a referência.
- Grave metade em silêncio (`ambiente=silencio`) e metade com barulho de mesa: conversa ao fundo, dados rolando, música (`ambiente=mesa`). Assim dá para medir onde o áudio quebra.
- Se possível, peça a outra pessoa para gravar algumas frases (`locutor=amigo`), para ter variação de sotaque.
- O gravador do celular serve. Fale normalmente, sem soletrar os termos.

## O dicionário de correções (`data/vocab/correcoes.csv`)

Cada linha é um erro real do Whisper e a forma correta. Preencha a coluna `origem` com o nome da amostra onde o erro apareceu (por exemplo `a03.wav`).
O `scripts/eval_wer.py` mede o dicionário com **leave-one-out**: ao avaliar `a03.wav`, as correções que vieram de `a03.wav` são ignoradas.
Assim o resultado mostra se o dicionário generaliza para frases novas, e não apenas se ele decorou as frases de teste.
Correções com `origem` vazia são tratadas como genéricas, e o script avisa que elas podem inflar o resultado.
