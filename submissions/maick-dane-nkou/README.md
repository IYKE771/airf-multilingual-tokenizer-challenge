# Maick Dane Nkou

## Status: corrected training notebook, replacement model pending

**The committed `tokenizer.json` is still the legacy `c41-lower-alph500-b3`
artifact. It is not lossless. Do not submit it as the corrected model.**

The official corpus could not be downloaded in the repair environment because
of a TLS connection failure to Hugging Face. No competition model was retrained
there, and no new validation score is claimed. The notebook's disposable test
models are never exported as submissions.

## Why the old score was misleading

The legacy model uses BPE, NFC + lowercase, WhitespaceSplit, and ByteFallback.
Lowercasing loses case and this pre-tokenizer/decoder combination loses whitespace.
Byte fallback avoids unknown tokens but cannot recover discarded information.

The supplied official-checker output on 24,000 validation rows reports:

| Component | Legacy result |
| --- | ---: |
| Base score (mean ha/sw/yo/am) | approximately 1.7440 |
| English/French guardrail penalty | 0.0000 |
| Reconstruction penalty | 2.9986 |
| **Full validation score** | **4.7426** |
| Rows not reconstructed exactly | 23,989 / 24,000 |

The old 1.7440 figure was **not** the full score under reconstruction-aware
scoring. Passing the file-validity checks does not remove this penalty.
These are supplied results, not a new evaluation of the hidden test set.

## Corrected recipe

`notebook.ipynb` now trains `lossless-bytelevel-b3` from scratch:

- `tokenizers==0.22.1`, BPE, vocabulary 10,000, minimum frequency 5;
- no normalizer: preserves case, diacritics, and original Unicode representation;
- `ByteLevel(add_prefix_space=False, use_regex=True)` pre-tokenizer;
- matching `ByteLevel` decoder;
- all 256 byte-alphabet symbols included at training, no special tokens;
- official `train` split only, balanced round-robin with ha/sw/yo/am repeated x3;
- dataset `Similoluwa/african-multilingual-tokenizer-challenge` @ `v1.0.0`.

No pretrained tokenizer, published vocabulary/merge table, or external corpus
is used. Validation and synthetic regression examples never enter the final
trainer. Changing only the old model's decoder is not a valid substitute for
retraining.

## Run and replace the submission

1. Open this `notebook.ipynb` in Google Colab and run all cells (CPU is sufficient).
2. The notebook downloads the official checker at commit
   `75578f2400c39b1f8e31ce7e7104b37fbc470d11`, verifies its SHA-256, and loads
   that exact module. It never silently falls back to a stale `utils.py`.
3. It loads 240,000 training and 24,000 validation rows, trains a fresh model,
   saves/reloads it, and checks exact reconstruction of every validation row.
4. It evaluates with the pinned official checker, including **both** guardrail
   and reconstruction penalties. Any reconstruction loss or unknown-token
   emission blocks export even if the official file-validity flag is true.
5. On success, retrieve `tokenizer.json`, `metadata.yml` and `README.md` from
   `artifacts/lossless-bytelevel-b3/export/maick-dane-nkou/` (the notebook offers
   downloads in Colab). Replace the three corresponding files in this team
   directory and include the corrected notebook as `notebook.ipynb`.
6. Review the generated **measured** validation results before submitting.
   No particular fertility/guardrail score is guaranteed; BPE merge ties may
   differ between training runs. Always check each generated artifact.

Reports, helper modules, dataset caches, and test models belong under ignored
`artifacts/`, not in the submission directory. Only the four permitted team
files should be submitted. The original training notebook and artifact remain
available in Git at `6a10f8b5db317e15ba14be6232440f72a4f6bae1` for provenance.

## Regression coverage

The notebook includes executable tests for all six languages, NFC/NFD text,
case, punctuation, repeated/boundary whitespace, tabs, CR/LF, emoji, literal
`[UNK]`/`[CLS]` text, controls, and unseen Unicode scalars. It verifies the
serialized model and rejects deliberately broken lowercasing, prefix-space,
WhitespaceSplit, and decoder configurations. Export-gate tests cover lossy
but technically valid reports, missing penalties, incomplete evaluation,
unknown tokens, and non-finite scores.

Exact round-trip checks preserve every character, which is stricter than the
pinned official reconstruction comparison (which tolerates NFC and boundary
whitespace differences). The official rules may evolve after the checker pin.
