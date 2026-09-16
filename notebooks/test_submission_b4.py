"""Offline tests for automatic train/evaluate/export, not full-corpus score evidence."""
import ast
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'submissions/maick-dane-nkou/notebook.ipynb'
MODEL = ROOT / 'submissions/maick-dane-nkou/tokenizer.json'
os.environ.setdefault('MPLBACKEND', 'Agg')


def cell(index):
    return ''.join(json.loads(NOTEBOOK.read_text())['cells'][index]['source'])


def definitions(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    return compile(tree, 'notebook definitions', 'exec')


@pytest.fixture
def scope(tmp_path, monkeypatch):
    pytest.importorskip('datasets')
    pytest.importorskip('matplotlib')
    import tokenizers
    from starter import utils
    monkeypatch.chdir(tmp_path)
    ctx = {'TOKENIZERS_VERSION': '0.22.1', 'tokenizers': tokenizers, 'official_utils': utils}
    exec(cell(4), ctx)
    exec(cell(10), ctx)
    exec(cell(12), ctx)  # Real disposable regression training; never a submitted model.
    exec(definitions(cell(19)), ctx)
    yield ctx
    ctx['plt'].close('all')


def report(scope, penalty=0.0):
    # Synthetic, internally consistent metrics to test export plumbing at <5% margin.
    score = 1.937123
    budget = 1.15 * score
    fertility = dict.fromkeys(scope['LANGUAGES'], score)
    fertility.update(en=2.0, fr=budget * .982 if penalty == 0 else budget + penalty)
    return {'valid': True, 'rows': 24000, 'vocab_size': 10000, 'score': score + penalty,
            'fertility': fertility, 'penalised': fertility.copy(),
            'unknown_rate': dict.fromkeys(scope['LANGUAGES'], 0.0),
            'guardrail_budget': budget, 'guardrail_penalty': penalty,
            'reconstruction': 1.0, 'lossy_rows': 0, 'reconstruction_penalty': 0.0}


def prepare(scope):
    path = scope['RUN_DIR'] / 'candidate/tokenizer.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(MODEL.read_bytes())
    data = scope['pd'].DataFrame({'language': [l for l in scope['LANGUAGES'] for _ in range(4000)],
                                  'text': ['  Unchanged é e\u0301\ttext.\n'] * 24000})
    data.attrs['dataset_fingerprint'] = 'SYNTHETIC-VALIDATION-NOT-OFFICIAL'
    train = types.SimpleNamespace(attrs={'dataset_fingerprint': 'SYNTHETIC-TRAIN-NOT-OFFICIAL'})
    class Train:
        attrs = train.attrs
        def __len__(self):
            return 240000
    scope.update(candidate_path=path, validation=data, train=Train(), training_seconds=0.01,
                 trained_config={'recipe': 'weights-b4', 'boost': dict(scope['BOOST']),
                                 'min_frequency': 5, 'vocab_size': 10000})
    return path


def test_schema_and_automatic_train_defaults():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook['nbformat'] == 4
    for c in notebook['cells']:
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), str(NOTEBOOK), 'exec')
            assert not c['outputs'] and c['execution_count'] is None
    assert 'RECIPE = "weights-b4"' in cell(4)
    assert 'tokenizer = train_final(corpus_iterator(train_by_lang))' in cell(13)
    assert 'profile_submission(candidate_path, data=validation' in cell(15)
    source = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    for removed in ('files.upload', 'review_reports.zip', 'official_report.json',
                    'validation_report.json', 'EXPORT_CANDIDATE', 'REVIEW ONLY'):
        assert removed not in source


def test_b4_train_only_iterator(scope):
    rows = {l: [f'TRAIN-{l}'] for l in scope['LANGUAGES']}
    texts = list(scope['corpus_iterator'](rows))
    assert len(texts) == 18
    assert scope['BOOST'] == {'ha': 4, 'sw': 4, 'yo': 4, 'am': 4}
    for l in scope['LANGUAGES']:
        assert texts.count(f'TRAIN-{l}') == (1 if l in ('en', 'fr') else 4)


def test_evaluate_and_auto_download_three_files_without_reports(scope, monkeypatch):
    path = prepare(scope)
    expected = report(scope)
    calls = []
    def profile(candidate, *, data, repeats):
        assert candidate == path and len(data) == 24000
        return expected
    scope['profile_submission'] = profile
    colab = types.ModuleType('google.colab')
    colab.files = types.SimpleNamespace(download=lambda p: calls.append(Path(p)))
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    exec(cell(15), scope)
    exec(cell(19), scope)
    assert [p.name for p in calls] == ['tokenizer.json', 'metadata.yml', 'README.md']
    out = scope['export_dir']
    assert sorted(p.name for p in out.iterdir()) == ['README.md', 'metadata.yml', 'tokenizer.json']
    assert (out / 'tokenizer.json').read_bytes() == path.read_bytes()
    assert '1.937123' in (out / 'README.md').read_text()
    assert '1.937123' in (out / 'metadata.yml').read_text()
    assert 'SYNTHETIC-TRAIN-NOT-OFFICIAL' in (out / 'README.md').read_text()
    assert '1.800000000%' in (out / 'README.md').read_text()
    assert hashlib.sha256(path.read_bytes()).hexdigest() in (out / 'README.md').read_text()
    assert not list(scope['RUN_DIR'].glob('*report*'))


def test_penalty_displayed_before_automatic_export_block(scope):
    prepare(scope)
    scope['profile_submission'] = lambda *a, **k: report(scope, penalty=.02)
    exec(cell(15), scope)
    assert scope['report']['score'] == pytest.approx(1.957123)
    with pytest.raises(ValueError, match='nonzero official penalty'):
        exec(cell(19), scope)
    assert not (scope['RUN_DIR'] / 'export').exists()


@pytest.mark.parametrize('mutation', ['hash', 'lossy', 'rows', 'unk', 'config', 'provenance', 'directory'])
def test_export_rejects_invalid_candidate_before_writing(scope, mutation):
    path = prepare(scope)
    measured = report(scope)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    output = scope['RUN_DIR'] / 'export'
    provenance = {'train_rows': 240000, 'validation_rows': 24000, 'train_fingerprint': 'fixture',
                  'validation_fingerprint': 'fixture', 'training_seconds': .01, 'tokenizers_version': '0.22.1'}
    if mutation == 'hash':
        digest = '0' * 64
    elif mutation == 'lossy':
        from tokenizers import normalizers
        tok = scope['Tokenizer'].from_file(str(path))
        tok.normalizer = normalizers.Lowercase()
        tok.save(str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    elif mutation == 'rows':
        measured['rows'] = 18
    elif mutation == 'unk':
        measured['unknown_rate']['ha'] = .1
    elif mutation == 'config':
        scope['trained_config']['boost'] = {'ha': 2}
    elif mutation == 'provenance':
        provenance['train_rows'] = 18
    elif mutation == 'directory':
        output = scope['RUN_DIR'] / 'submissions/team'
    with pytest.raises(ValueError):
        scope['export_submission'](path, measured, digest, scope['validation'].text.tolist(),
                                   output, scope['trained_config'], provenance)
    assert not output.exists()
