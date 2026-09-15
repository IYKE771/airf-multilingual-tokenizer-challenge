"""Offline regression tests for the standalone exploration notebook.

Synthetic metrics exercise orchestration/selection, NOT competition performance.
Run: python -m pytest tests notebooks/test_optimisation_b4.py
"""
import copy
import hashlib
import io
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/optimisation_b4.ipynb"
MODEL = ROOT / "submissions/maick-dane-nkou/tokenizer.json"


def source(tag):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return next("".join(c["source"]) for c in notebook["cells"] if tag in c["metadata"].get("tags", []))


@pytest.fixture
def scope(tmp_path):
    ctx = {"__name__": "notebook_test"}
    exec(source("config"), ctx)
    exec(source("worker"), ctx)
    module = types.ModuleType("test_cpu_worker")
    exec(ctx["WORKER_SOURCE"], module.__dict__)
    ctx.update(worker=module, sha256_file=module.sha256_file, atomic_json=module.atomic_json,
               WORK_DIR=tmp_path / "runtime", WORKER_PATH=tmp_path / "cpu_worker.py",
               TRAIN_PATH=tmp_path / "train.jsonl", VALIDATION_PATH=tmp_path / "validation.jsonl",
               CHECKER_PATH=tmp_path / "official_utils.py", SAVE_ROOT=tmp_path / "saved")
    ctx["WORK_DIR"].mkdir()
    ctx["SAVE_ROOT"].mkdir()
    ctx["WORKER_PATH"].write_text(ctx["WORKER_SOURCE"], encoding="utf-8")
    ctx["TRAIN_PATH"].write_text('["en","TRAIN ONLY"]\n', encoding="utf-8")
    ctx["VALIDATION_PATH"].write_text('["en","VALIDATION ONLY"]\n', encoding="utf-8")
    exec(source("search-functions"), ctx)
    return ctx


def metrics(ctx, score=1.94, headroom=.0179, unk=0.0, lossy=0):
    """Fabricated but internally consistent official-format report."""
    fertility = dict.fromkeys(ctx["LANGUAGES"], score)
    budget = 1.15 * score
    fertility.update(en=budget * (1 - headroom - .02), fr=budget * (1 - headroom))
    unknown = dict.fromkeys(ctx["LANGUAGES"], unk)
    penalised = {l: fertility[l] + 100 * unk for l in ctx["LANGUAGES"]}
    guard = sum(max(0, fertility[l] - budget) for l in ("en", "fr"))
    reconstruction = 1 - lossy / 24_000
    return {"official": {
        "valid": True, "rows": 24_000, "vocab_size": 10_000, "tokenizers_version": "0.22.1",
        "fertility": fertility, "unknown_rate": unknown, "penalised": penalised,
        "guardrail_budget": budget, "guardrail_penalty": guard,
        "reconstruction": reconstruction, "reconstruction_penalty": 3 * (1 - reconstruction),
        "lossy_rows": lossy, "score": score + 100 * unk + guard + 3 * (1 - reconstruction),
    }, "strict_failures": lossy}


def context(ctx):
    return {"train": {"sha256": ctx["sha256_file"](ctx["TRAIN_PATH"])},
            "validation": {"sha256": ctx["sha256_file"](ctx["VALIDATION_PATH"])}}


def fake_worker(ctx, calls):
    """Fake worker IO only; never reports actual BPE search measurements."""
    def run(arguments, log_path):
        calls.append(arguments)
        Path(log_path).write_text("SYNTHETIC TEST ONLY\n", encoding="utf-8")
        if arguments[0] == "train":
            _, train_path, config_path, model_path = arguments
            assert train_path == ctx["TRAIN_PATH"]
            assert ctx["VALIDATION_PATH"] not in arguments
            config = json.loads(Path(config_path).read_text())
            Path(model_path).write_bytes(MODEL.read_bytes())
            ctx["atomic_json"](Path(model_path).parent / "training.json", {
                "config": config, "train_sha256": ctx["sha256_file"](train_path),
                "tokenizer_sha256": ctx["sha256_file"](model_path), "training_seconds": 0.0,
                "weighted_rows": 40_000 * sum(config["quarter_units"].values()) // 4,
            })
        else:
            _, model_path, validation_path, _, output = arguments
            assert validation_path == ctx["VALIDATION_PATH"]
            name = Path(model_path).parent.name
            # am5 improves but loses margin; yo5 improves and preserves it.
            score = 1.94 if name == "b4-reference" else (1.92 if name == "am5" else 1.93)
            report = metrics(ctx, score, .0178 if name == "am5" else .0179)
            report.update(tokenizer_sha256=ctx["sha256_file"](model_path),
                          validation_sha256=ctx["sha256_file"](validation_path), checker_sha256=ctx["CHECKER_SHA256"])
            ctx["atomic_json"](Path(output) / "official_report.json", report["official"])
            ctx["atomic_json"](Path(output) / "evaluation.json", report)
    return run


def test_schema_syntax_and_fixed_grid(scope):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
    compile(scope["WORKER_SOURCE"], "worker", "exec")
    configs = scope["CONFIGS"]
    assert len(configs) == len({c["name"] for c in configs}) == 6
    assert configs[0]["name"] == "b4-reference"
    assert configs[0]["quarter_units"] == dict(zip(scope["LANGUAGES"], (4, 4, 16, 16, 16, 16)))
    assert all(c["vocab_size"] == 10000 and c["min_frequency"] == 5 for c in configs)
    assert "DOWNLOAD_REPORTS = False" in source("review-bundle")
    assert "DOWNLOAD_BEST_TOKENIZER = False" in source("review-bundle")


def test_weighted_iterator_exact_fractional_and_baseline_order(scope):
    rows = [(l, f"{l}-{i}") for i in range(8) for l in scope["LANGUAGES"]]
    for config in scope["CONFIGS"]:
        result = list(scope["worker"].weighted_texts(iter(rows), config["quarter_units"]))
        counts = scope["Counter"](s.split("-")[0] for s in result)
        assert counts == {l: 8 * u // 4 for l, u in config["quarter_units"].items()}
        assert all(s in result for _, s in rows)
    baseline = list(scope["worker"].weighted_texts(rows, scope["CONFIGS"][0]["quarter_units"]))
    assert baseline == [s for l, s in rows for _ in range(1 if l in ("en", "fr") else 4)]
    french = scope["CONFIGS"][-1]["quarter_units"]
    result = list(scope["worker"].weighted_texts(rows, french))
    assert result.count("fr-3") == result.count("fr-7") == 2
    assert result.count("fr-0") == result.count("fr-1") == 1
    with pytest.raises(ValueError):
        list(scope["worker"].weighted_texts(rows, {"en": 4}))


def test_real_disposable_bpe_roundtrip_and_lossy_mutations(scope):
    exec(source("regression"), scope)


def test_exact_measured_margin_not_rounded_or_five_percent(scope):
    reference = metrics(scope)
    assert scope["selection_reasons"](metrics(scope, 1.93, .0179), reference) == []
    assert "marge inférieure au b4 réévalué" in scope["selection_reasons"](metrics(scope, 1.92, .0178), reference)
    assert "pas de baisse du score" in scope["selection_reasons"](metrics(scope), reference)
    assert "pas de baisse du score" in scope["selection_reasons"](metrics(scope, 1.95, .03), reference)


def test_all_penalties_visible_but_ineligible(scope):
    reference = metrics(scope)
    for candidate in (metrics(scope, 1.90, -.01), metrics(scope, 1.90, .03, unk=.0001), metrics(scope, 1.90, .03, lossy=1)):
        measured = scope["metric_summary"](candidate)
        assert measured["score"] == candidate["official"]["score"]
        assert scope["selection_reasons"](candidate, reference)
    broken = metrics(scope, 1.90, -.01)
    broken["official"]["score"] -= broken["official"]["guardrail_penalty"]
    with pytest.raises(ValueError, match="arithmetic"):
        scope["metric_summary"](broken)


@pytest.mark.parametrize("key,value", [("valid", False), ("rows", 18), ("score", float("nan")),
                                      ("vocab_size", 9999), ("reconstruction", .99),
                                      ("tokenizers_version", "0.21.0"), ("guardrail_budget", 0)])
def test_incomplete_invalid_and_inconsistent_reports_rejected(scope, key, value):
    report = metrics(scope)
    report["official"][key] = value
    with pytest.raises(ValueError):
        scope["metric_summary"](report)


def test_six_candidate_run_resume_and_saved_ranking(scope):
    calls = []
    scope["run_worker"] = fake_worker(scope, calls)
    session = scope["SAVE_ROOT"]
    ctx = context(scope)
    history, best = scope["run_search"](scope["CONFIGS"], ctx, session)
    assert len(history) == 6 and best["config"]["name"] == "yo5"
    assert len([c for c in calls if c[0] == "train"]) == 6
    assert len([c for c in calls if c[0] == "evaluate"]) == 6
    saved = json.loads((session / "search_history.json").read_text())
    assert saved["review_only"] is True
    assert saved["minimum_headroom"] == pytest.approx(.0179)
    assert saved["best"] == "yo5"
    assert (session / "comparison.csv").is_file()
    for entry in history:
        directory = Path(entry["directory"])
        assert (directory / "official_report.json").is_file()
        # Poisoning saved score must not affect fresh evaluations on resume.
        (directory / "evaluation.json").write_text('{"score": -1}')
    calls.clear()
    _, resumed_best = scope["run_search"](scope["CONFIGS"], ctx, session)
    assert resumed_best["config"]["name"] == "yo5"
    assert [c[0] for c in calls] == ["evaluate"] * 6


def test_failed_baseline_stops_before_alternatives(scope):
    calls = []
    def fail(config, *args):
        calls.append(config["name"])
        raise RuntimeError("synthetic OOM")
    scope["run_candidate"] = fail
    with pytest.raises(RuntimeError, match="Baseline failed"):
        scope["run_search"](scope["CONFIGS"], context(scope), scope["SAVE_ROOT"])
    assert calls == ["b4-reference"]
    saved = json.loads((scope["SAVE_ROOT"] / "search_history.json").read_text())
    assert saved["best"] is None and saved["candidates"][0]["status"] == "failed"


def test_alternative_failure_logged_and_later_candidates_run(scope):
    calls = []
    original = fake_worker(scope, calls)
    def fail_one(args, log):
        if args[0] == "train" and "am5" == Path(args[-1]).parent.name:
            Path(log).write_text("SYNTHETIC FAILURE")
            raise RuntimeError("synthetic OOM")
        original(args, log)
    scope["run_worker"] = fail_one
    history, best = scope["run_search"](scope["CONFIGS"], context(scope), scope["SAVE_ROOT"])
    assert len(history) == 6 and history[1]["status"] == "failed"
    assert best["config"]["name"] == "yo5"
    assert "SYNTHETIC FAILURE" in (scope["SAVE_ROOT"] / "am5/training.log").read_text()


def test_no_admissible_gain_retains_reference(scope):
    baseline = {**metrics(scope), "status": "reference", "config": scope["CONFIGS"][0]}
    other = {**metrics(scope, 1.92, .01), "status": "measured", "config": scope["CONFIGS"][1]}
    assert scope["choose_best"]([baseline, other]) is baseline


def test_cache_corruption_provenance_and_incomplete_marker(scope):
    calls = []
    scope["run_worker"] = fake_worker(scope, calls)
    config, ctx, session = scope["CONFIGS"][0], context(scope), scope["SAVE_ROOT"]
    entry = scope["run_candidate"](config, ctx, session)
    directory = Path(entry["directory"])
    expected = {"context": ctx, "config": config}
    assert scope["verify_saved_candidate"](directory, expected)
    wrong = copy.deepcopy(expected)
    wrong["config"]["min_frequency"] = 2
    with pytest.raises(ValueError, match="provenance"):
        scope["verify_saved_candidate"](directory, wrong)
    model = directory / "tokenizer.json"
    model.write_bytes(model.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        scope["verify_saved_candidate"](directory, expected)
    (directory / "manifest.json").unlink()
    assert scope["verify_saved_candidate"](directory, expected) is None
    calls.clear()
    scope["run_candidate"](config, ctx, session)
    assert [c[0] for c in calls] == ["train", "evaluate"]


def test_changed_corpus_blocks_search(scope):
    ctx = context(scope)
    scope["TRAIN_PATH"].write_text("changed")
    with pytest.raises(ValueError, match="changed after provenance"):
        scope["run_search"](scope["CONFIGS"], ctx, scope["SAVE_ROOT"])


def test_train_and_eval_reject_reduced_corpora(scope):
    path = scope["WORK_DIR"] / "config.json"
    scope["atomic_json"](path, scope["CONFIGS"][0])
    with pytest.raises(ValueError, match="Full official train"):
        scope["worker"].train_job(scope["TRAIN_PATH"], path, scope["WORK_DIR"] / "model.json")
    with pytest.raises(ValueError, match="Full official validation"):
        scope["worker"].evaluate_job(MODEL, scope["VALIDATION_PATH"], scope["CHECKER_PATH"], scope["WORK_DIR"])
    scope["CHECKER_PATH"].write_text("untrusted helper")
    with pytest.raises(ValueError, match="SHA-256"):
        scope["worker"].load_checker(scope["CHECKER_PATH"])


def test_data_serialization_preserves_strings_and_train_order(scope, monkeypatch):
    # Generated fixture rows satisfy counts only to test IO; not official dataset evidence.
    def text(l, i):
        return f"  {l} {i} é e\u0301\t\n "
    class FakeDataset:
        _fingerprint = "SYNTHETIC-NOT-OFFICIAL"
        def __init__(self, split):
            self.n = 40_000 if split == "train" else 4_000
        def __iter__(self):
            for l in scope["LANGUAGES"]:
                for i in range(self.n):
                    yield {"language": l, "text": text(l, i)}
    seen = []
    def load(identifier, *, revision, split, cache_dir):
        seen.append(split)
        assert identifier == scope["DATASET_ID"] and revision == "v1.0.0"
        return FakeDataset(split)
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load))
    exec(source("data-functions"), scope)
    train_info = scope["prepare_split"]("train", scope["TRAIN_PATH"])
    validation_info = scope["prepare_split"]("validation", scope["VALIDATION_PATH"])
    assert seen == ["train", "validation"]
    assert train_info["rows"] == 240_000 and validation_info["rows"] == 24_000
    rows = scope["worker"].read_rows(scope["TRAIN_PATH"])
    assert [next(rows) for _ in range(12)] == [(l, text(l, i)) for i in range(2) for l in scope["LANGUAGES"]]
    rows.close()
    assert train_info["sha256"] == scope["sha256_file"](scope["TRAIN_PATH"])
    target = io.StringIO()
    scope["dump_row"](target, "fr", "  A\t\n e\u0301  ")
    assert json.loads(target.getvalue()) == ["fr", "  A\t\n e\u0301  "]
    with pytest.raises(ValueError, match="Seulement"):
        scope["prepare_split"]("test", scope["VALIDATION_PATH"])


def test_review_bundle_excludes_corpus_and_models(scope):
    scope["run_worker"] = fake_worker(scope, [])
    ctx = context(scope)
    history, best = scope["run_search"](scope["CONFIGS"], ctx, scope["SAVE_ROOT"])
    scope.update(history=history, best=best, CONTEXT=ctx, SESSION_DIR=scope["SAVE_ROOT"])
    scope["atomic_json"](scope["SESSION_DIR"] / "run_context.json", ctx)
    exec(source("review-bundle"), scope)
    with scope["zipfile"].ZipFile(scope["archive_path"]) as archive:
        assert "comparison.csv" in archive.namelist()
        assert not any(n.endswith("tokenizer.json") or n.endswith(".jsonl") for n in archive.namelist())
        assert len([n for n in archive.namelist() if n.endswith("official_report.json")]) == 6


def test_subprocess_log_and_error_propagation(scope):
    scope["WORKER_PATH"].write_text("import sys\nprint('worker fixture', flush=True)\nsys.exit(int(sys.argv[1]))\n")
    log = scope["WORK_DIR"] / "worker.log"
    scope["run_worker"]([0], log)
    assert "worker fixture" in log.read_text()
    with pytest.raises(RuntimeError, match="Worker failed"):
        scope["run_worker"]([7], log)
