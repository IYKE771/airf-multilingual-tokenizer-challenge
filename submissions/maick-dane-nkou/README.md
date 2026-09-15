# Maick Dane Nkou — lossless byte-level BPE

## Current submitted artifact

The committed `tokenizer.json` is the **new lossless model uploaded by the
participant**, not the former lowercase/WhitespaceSplit model.

- Uploaded in commit `07264e0b0a20d3e179f5f4ecad5448505e2dbd1d`.
- SHA-256: `9cc713ef8c779d4b657b10afbb4a1843751e8f69ad419342a95961efebd52edf`.
- 10,000 vocabulary entries; `tokenizers==0.22.1`.
- No normalizer, special tokens, or post-processor.
- `ByteLevel(add_prefix_space=False, use_regex=True)` pre-tokenizer and
  matching `ByteLevel` decoder, full 256-symbol byte alphabet.
- BPE trained from scratch on the official train split only, minimum frequency
  5; ha/sw/yo/am repeated x3, English/French x1, balanced round-robin.
- Dataset: `Similoluwa/african-multilingual-tokenizer-challenge` @ `v1.0.0`.
- No pretrained tokenizer, third-party vocabulary/merge table, or external corpus.

The original lossless training notebook is preserved at commit
`2805e5a52b1f9a859e871387ec71b952914f6ab3`. It can retrain the reference recipe
without downloading any pretrained model. BPE merge ties may differ between
runs; always evaluate each regenerated artifact.

## Participant-reported validation results

The participant supplied these results from the corrected Colab notebook on
**24,000 official validation rows**. They are not hidden-test scores and were
not independently remeasured on the full dataset in the development workspace.

| Language | Tokens/word | UNK rate |
| --- | ---: | ---: |
| English | 2.031 | 0.0000 |
| French | 2.206 | 0.0000 |
| Hausa | 1.707 | 0.0000 |
| Swahili | 1.889 | 0.0000 |
| Yoruba | 2.049 | 0.0000 |
| Amharic | 2.449 | 0.0000 |

- **Full validation score: 2.023252** (2.0233 when rounded to four decimals).
- Guardrail penalty: **0.000000**; reported headroom: 5.2%.
- Reconstruction penalty: **0.000000**.
- Strict reconstruction: **100%**, every original character preserved.

The former model's 1.7440 was a base-only score. Its full score including the
reconstruction penalty was 4.7426. Compare **full scores**, not the old base-only
number against a reconstruction-aware score.

Local checks confirm that the uploaded file loads, stays within the vocabulary
limit and reconstructs the six smoke-test languages. This is not a substitute
for evaluation on all validation rows.

## Optimization notebook

`notebook.ipynb` now prepares a **bounded search**, not an already proven lower
score. The corpus remains unreachable from the development workspace because
of a TLS connection failure to Hugging Face; run the search in Colab.

The search first loads the exact uploaded reference from its pinned commit
(or the matching local file), verifies its SHA-256 and **remeasures it** using
the same checker and validation data as the candidates. Its candidates are:

1. Two cheap boundary-only adaptations of this participant's own reference:
   combining-mark-aware boundaries and punctuation-attached word boundaries.
   The vocabulary/merges are unchanged, and every separator is retained.
2. Two fresh trainings with these alternative boundary strategies and x3 weights.
3. Three fresh trainings using the best eligible boundary strategy: scored
   languages x2, scored languages x4, and ha/sw x2 with yo/am x4.
4. Two fresh trainings with the winning mode/weights and minimum frequencies
   2 and 10.

All three boundary strategies use **no normalization**, the complete byte
alphabet and the matching ByteLevel decoder. The custom `Split` steps use
`behavior="isolated"`, never the lossy `WhitespaceSplit` configuration.
Synthetic regression data is only used for disposable tests, never for a
competition model. Validation is only used to evaluate/select candidates,
not to train their vocabulary or merges.

### Promotion rules

- Pass the official validity contract and use 10,000 vocabulary entries.
- Strict exact reconstruction of every original validation row, zero UNK.
- Zero official reconstruction penalty and zero English/French guardrail penalty.
- At least 5% context-language guardrail headroom (a heuristic, not a hidden-test guarantee).
- A strictly lower **full official score** than the current best.

The existing reference is retained on ties, failures, or lack of improvement.
No candidate replaces the committed tokenizer automatically. Repeated tuning
can overfit validation; an improvement there does not guarantee an improvement
on the hidden split.

## Run and export

1. Open this notebook in Google Colab and run all cells. Seven full trainings
   take substantially longer than the previous single training run.
2. The notebook downloads and hash-verifies the official checker at commit
   `75578f2400c39b1f8e31ce7e7104b37fbc470d11`. It never silently uses a stale
   `utils.py`. The checker pin records the rules version; organizers may update it.
3. Inspect `artifacts/lossless-bpe-search/search_results.csv` and
   `search_history.json` for the candidate comparisons and any failures.
4. The winner is hash-checked, reloaded, and re-evaluated before export.
5. Retrieve `tokenizer.json`, `metadata.yml`, and `README.md` from
   `artifacts/lossless-bpe-search/export/maick-dane-nkou/` and include the notebook
   as `notebook.ipynb`. The generated documentation records the actual selected
   recipe, provenance, file hash, and measured full score.
6. If the notebook says the reference was retained, **no improvement was found**;
   keep the current artifact instead of claiming a lower score.

Datasets, intermediate models, helpers, and detailed reports stay under ignored
`artifacts/`; only the four permitted team files belong in this directory.
The notebook includes executable regression tests for Unicode/whitespace
round-trips, all boundary modes, and rejection of lossy, worse, tied, invalid,
or insufficiently balanced candidate scores.
