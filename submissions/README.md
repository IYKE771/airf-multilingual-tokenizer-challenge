# Maick Dane Nkou — lossless BPE

Official train only; no pretrained tokenizer or external corpus.

## Selected recipe and provenance

- Candidate: `weights-yo-am4`; origin: `trained-from-scratch`
- Training boundary mode: `space_word`
- Inference boundary mode: `space_word`
- Language repeats: `{"am": 4, "ha": 2, "sw": 2, "yo": 4}` (others x1)
- BPE, vocabulary 10000, min_frequency=5
- tokenizers==0.22.1; no normalizer, special tokens, or post-processor
- Full byte alphabet; ByteLevel decoder; no artificial prefix space
- Dataset: `Similoluwa/african-multilingual-tokenizer-challenge` @ `v1.0.0`
- Reference artifact from own training: `07264e0b0a20d3e179f5f4ecad5448505e2dbd1d`
- Uploaded/adapted references reuse only the participant's own train-only vocabulary.
- From-scratch candidates train fresh vocabularies on train; validation is used only for selection.

## Measured validation results (24,000 rows; not hidden-test scores)

| Language | Tokens/word | UNK rate |
| --- | ---: | ---: |
| en | 2.028270 | 0.000000 |
| fr | 2.137162 | 0.000000 |
| ha | 1.751356 | 0.000000 |
| sw | 1.973766 | 0.000000 |
| yo | 1.839887 | 0.000000 |
| am | 2.287749 | 0.000000 |

- Reference full score: 2.023252
- Selected base score: 1.963189
- Guardrail penalty: 0.000000
- Reconstruction penalty: 0.000000
- **Selected full score: 1.963189**
- Strict exact reconstruction: 24,000 / 24,000 rows (100%)
- Official checker commit: `75578f2400c39b1f8e31ce7e7104b37fbc470d11`
- Tokenizer SHA-256: `1519895eace8680d2752b333f5efd82f80ade21bcd6210704ec17de55dafe035`

## Reproduce

Run notebook.ipynb end-to-end in Colab. It loads the pinned own reference, trains
seven candidates on official train only, tries two boundary adaptations, and compares
the full validation score including penalties. It retains the reference if no eligible
candidate improves it. Byte-level boundary definitions are in the notebook.
The original reference training notebook is at commit 2805e5a52b1f9a859e871387ec71b952914f6ab3.
BPE merge ties may differ across retraining runs. Evaluate every generated artifact.
