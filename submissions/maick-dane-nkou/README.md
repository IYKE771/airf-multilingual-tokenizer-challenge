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

## Notebook: train, evaluate and automatically export b4

The committed tokenizer and results above still describe **weights-yo-am4**;
this notebook update alone does not replace its bytes or metadata.

Run `notebook.ipynb` end-to-end on CPU to train **weights-b4** from scratch:
ha/sw/yo/am x4, en/fr x1, the same lossless space_word pipeline, 10,000 entries,
minimum frequency 5. It uses the full 240,000-row official train split, then
calls the hash-pinned official checker on all 24,000 validation rows.
No uploaded tokenizer is needed and no diagnostic archive is produced.

The full score includes all official penalties. The historical 5% headroom
policy is informational, not an export gate. After full validation, zero UNK,
zero official penalties and strict reconstruction checks, Colab automatically
downloads `tokenizer.json`, `metadata.yml` and `README.md`. A failed check stops
export, without hiding the checker output. Allow multiple downloads if prompted.

Files are written under `artifacts/submission-weights-b4/export/maick-dane-nkou/`,
not over the repository submission. The generated README includes the actual
score, model SHA-256, data fingerprints and training time. There is no review-only
switch or request to upload reports. Downloading is not a GitHub submission.

The user's previous b4 runs measured 1.939666974 with 1.787199439% headroom.
BPE merge ties can vary: a fresh score and file hash are measured, not assumed.
To reproduce the currently committed recipe instead, set
`RECIPE = "weights-yo-am4"` before running all cells.
The separate optimization notebook is unchanged. No new full training or
validation result is claimed by this code update.
