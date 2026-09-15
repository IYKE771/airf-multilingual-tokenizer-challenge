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

`notebook.ipynb` now follows the official starter's structure: Getting Started,
Load the Dataset, Train the Final Tokenizer, Score the Tokenizer, Compare with
Previous Experiments, Prepare Your Submission, Final Checklist, and References.
It trains **only the selected `weights-yo-am4` recipe once**, from scratch on the
public train split, rather than rerunning the seven-candidate search. The recipe,
training weights and byte-level boundaries are unchanged.

The original optimization notebook and submitted artifact are preserved at
commit `17d34954a86e9baabb46658478ac7a0160e3a04d`. The earlier reference training
notebook is at `2805e5a52b1f9a859e871387ec71b952914f6ab3`.

Run the final notebook end-to-end in Colab or a prepared local kernel. It installs
only missing dependencies, loads the official public train/validation splits,
inspects the training distribution, trains the chosen recipe, reloads the saved
model, and calls the pinned official `profile_submission` helper. The full score
includes reconstruction and English/French guardrail penalties. Strict round-trip
checks and our 5% guardrail-margin policy protect export; the margin is **not** an
official validity requirement.

Generated reports, caches and exports stay under ignored
`artifacts/final-weights-yo-am4/`, not inside the team directory, even when the
kernel starts in this directory. Review the newly measured results before
replacing the current model with the exported `tokenizer.json`, `metadata.yml`
and `README.md`. BPE merge ties may vary across retraining runs; the previous
score and hash are not promised for a fresh run.

**This notebook reorganization does not modify the submitted tokenizer or its
recorded validation results above.** No complete retraining or 24,000-row
re-evaluation was performed in the development workspace during this change.

## Pull request checklist

- Include only this team's four permitted files in the competition-entry diff.
- Keep `tokenizer.json`, `metadata.yml`, `notebook.ipynb` and `README.md` together
  in `submissions/maick-dane-nkou/`; no symlinks, archives or generated sidecars.
- Target `main` of the official AIMS repository.
- Do not change the workflows, starter notebook, evaluator, or leaderboard.

The inspected official validation workflow checks the team directory and
serialized tokenizer, then runs the evaluator contract tests. It does not
execute the participant notebook or require the starter's exact layout. A
`pull_request` affecting `submissions/**` triggers validation even if its source
branch is not named `submission`; only the fork's push trigger is restricted to
that name. GitHub approval/runner issues can still require organizer action.
