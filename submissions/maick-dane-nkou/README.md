# Maick Dane Nkou

Individual entry — tokenizer `c41-lower-alph500-b3`.

## Approach

- Model: BPE with `[UNK]` as unknown token; vocabulary 10,000 / 10,000
- Normalizer: NFC + lowercase
- Alphabet: limited to the 500 most frequent train characters (`limit_alphabet`); the characters left out are still covered by the byte fallback, so no `[UNK]` appears
- Pre-tokenizer: `WhitespaceSplit` — punctuation stays attached to its word, so no token is spent on isolated `,` `.` `)` …
- Byte fallback: yes — the 256 `<0xXX>` tokens cover every possible UTF-8 character, so
  no `[UNK]` can ever be emitted
- Training corpus: official `train` split only, balanced round-robin over the six
  languages with am/ha/sw/yo oversampled x3; no external corpus and no pre-trained tokenizer
- Post-processor: none | Decoder: ByteFallback
- Built with `tokenizers==0.22.1`

## Results (official validation split, official metric)

| Language | Fertility | UNK rate | Score |
|---|---:|---:|---:|
| English | 1.7828 | 0.000000 | 1.7828 |
| French | 1.9107 | 0.000000 | 1.9107 |
| Hausa | 1.4839 | 0.000000 | 1.4839 |
| Swahili | 1.6548 | 0.000000 | 1.6548 |
| Yoruba | 1.6535 | 0.000000 | 1.6535 |
| Amharic | 2.1838 | 0.000000 | 2.1838 |

- **Score (mean of ha, sw, yo, am): 1.7440** — baseline BPE 10k 2.0600, i.e. a gain of +0.3160 (15.3 %)
- Context guardrail EN/FR: **PASS** (budget 2.0056, en 1.7828, fr 1.9107)
- UNK emitted on validation: 0
- Jaccard robustness (quality): 0.9645

## Reproducibility

- Dataset: `Similoluwa/african-multilingual-tokenizer-challenge` @ `v1.0.0`
  (train 240,000 / validation 24,000, 40,000+4,000 per language)
- Training data: official `train` split only — no external corpus, no pretrained
  tokenizer, no third-party API
- Metric: official `fertility + 100 × unk_rate`, averaged over ha/sw/yo/am;
  guardrail `fertility(en,fr) ≤ 1.15 × mean(scored)` — all verified with the
  challenge's own evaluation code
- Environment: `tokenizers==0.22.1` (exact version required by the checker)
- Rebuild: running `notebook.ipynb` end-to-end retrains this exact tokenizer
  (config `c41-lower-alph500-b3`) and regenerates its reports

## Files

- `tokenizer.json` — the submitted tokenizer
- `metadata.yml` — team metadata
- `notebook.ipynb` — the notebook that built this tokenizer (same training as `notebooks/05_train_c41_final.ipynb`, without the GitHub publishing cells)
- `README.md` — this file
