# Maick Dane Nkou — lossless BPE

## Recipe

- weights-yo-am4: space_word boundaries, no normalizer or special tokens
- BPE 10,000; minimum frequency 5; full byte alphabet; ByteLevel decoder
- Official train only; balanced round-robin; ha/sw x2, yo/am x4, en/fr x1
- Dataset: `Similoluwa/african-multilingual-tokenizer-challenge` @ `v1.0.0`
- No pretrained tokenizer, published vocabulary/merge table or external corpus

## Measured validation results (24,000 rows; not hidden-test scores)

| Language | Tokens/word | UNK rate |
| --- | ---: | ---: |
| en | 2.028270 | 0.000000 |
| fr | 2.137162 | 0.000000 |
| ha | 1.751356 | 0.000000 |
| sw | 1.973766 | 0.000000 |
| yo | 1.839887 | 0.000000 |
| am | 2.287749 | 0.000000 |

- Base score: 1.963189
- Guardrail penalty: 0.000000
- Reconstruction penalty: 0.000000
- **Full score: 1.963189**
- Strict reconstruction: 100%
- Tokenizer SHA-256: `1519895eace8680d2752b333f5efd82f80ade21bcd6210704ec17de55dafe035`
- Official checker: `75578f2400c39b1f8e31ce7e7104b37fbc470d11`

## Reproduce

Run notebook.ipynb end-to-end. It follows the official starter flow and
trains the selected recipe once, from scratch on train only, then checks
the reloaded file on validation and exports measured results.
The original optimization notebook is preserved at commit 17d34954a86e9baabb46658478ac7a0160e3a04d.
BPE merge ties may vary across retraining runs; always evaluate the generated artifact.
