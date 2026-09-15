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

## Notebook: test the b4 candidate first

The committed model and measured results above remain **weights-yo-am4** and
are unchanged. The notebook now defaults to a **weights-b4 trial**: ha/sw/yo/am
x4, en/fr x1, with the same lossless space_word pipeline, 10,000 entries and
minimum frequency 5. That recipe previously reported 1.939667 on validation;
a fresh run must be evaluated rather than assumed to reproduce that score.

Run `notebook.ipynb` end-to-end to train the trial from scratch on official train
only, then run the pinned official checker on all 24,000 validation rows. The
extra 5% headroom policy is informational during this trial and does not stop
evaluation. Any official EN/FR penalty remains part of the measured full score.

Results are saved under `artifacts/trial-weights-b4/` as `official_report.json`
and `validation_report.json`. Export is disabled by default: inspect the result
before setting `EXPORT_CANDIDATE = True` in the last cell. Even when enabled,
export goes under artifacts and never automatically overwrites the submitted
model. Generated metadata records the actual trained preset and measured score.

To reproduce the **currently submitted** recipe instead, set
`RECIPE = "weights-yo-am4"` in the configuration cell before running all cells.
The original optimization notebook is preserved at commit
`17d34954a86e9baabb46658478ac7a0160e3a04d`; the prior final-recipe notebook is at
`24e3cd6bc30da2f72f1ff4697c387a048b2f317a`.
BPE merge ties may vary across retraining runs. No new b4 training or full
validation result is claimed by this notebook update.
