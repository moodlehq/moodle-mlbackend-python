import os
import re
import secrets
import json
from contextlib import contextmanager
from pprint import pprint
import time
import subprocess

import stash
import pytest
import testdata
import bz2
import traceback
import shutil

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, 'temp-data')

os.makedirs(DATA_DIR, exist_ok=True)
os.environ["MOODLE_MLBACKEND_PYTHON_DIR"] = DATA_DIR

# env setting to prevent deletion of (some) test models, for post
# mortum diagnostics.
KEEP_TEST_MODELS = os.environ.get("MOODLE_MLBACKEND_KEEP_TEST_MODELS")

# env setting to run normally skipped tests
RUN_SLOW_TESTS = os.environ.get("MOODLE_MLBACKEND_RUN_SLOW_TESTS")

BAD_UID = 'an unused unique ID that does not exist ' + secrets.token_hex(10)

TEST_MODEL = os.path.join(HERE, 'test-models', 'split-evaluation')


def unique_id(depth=1):
    """Create an ID that incorporates the calling function name, the
    timestamp, and some random characters.

    Depth indicates how far up the stacktrace to look for the function
    name.
    """
    f_name = traceback.extract_stack()[-depth].name
    now = int(time.time())
    token = secrets.token_urlsafe(6)

    return f'{f_name}-{now}-{token}'


@contextmanager
def temp_id_and_dir(depth=1):
    modelid = unique_id(depth + 1)
    directory = os.path.join(DATA_DIR, modelid)
    try:
        os.makedirs(directory)
        yield (modelid, directory)
    finally:
        shutil.rmtree(directory)


def extract_dataset(name):
    if '/' not in name:
        name = os.path.join(HERE, 'test-requests', name)

    data, headers, url = stash.load(name)
    boundary = stash.get_boundary(headers)
    parts = stash.split_body(data, boundary)
    for k, v in parts.items():
        h2, body = v
        cd = h2.get('Content-Disposition')
        if cd and cd.get('name') == '"dataset"':
            return body.decode()


@contextmanager
def temp_dataset(name):
    dataset = extract_dataset(name)
    with temp_id_and_dir(depth=2) as (modelid, directory):
        filename = os.path.join(directory, 'data.csv')
        with open(filename, 'w') as f:
            f.write(dataset)
        yield filename


def run(module, *params):
    """Transliterated from the PHP test.

    Runs the named submodule of the moodlemlbackend, with specified
    command line parameters, in a subprocess.

    Returns the subprocess result object.
    """
    cmd = ['python3',
           '-m', f'moodlemlbackend.{module}']

    cmd.extend(params)

    p = subprocess.run(cmd, capture_output=True)
    return p


def test_version():
    p = run('version')
    assert p.returncode == 0
    assert re.match(br'^\d+\.\d+\.\d+\n?', p.stdout) is not None


def test_training():
    with temp_dataset('split-evaluation-train.bz2') as dataset, \
         temp_id_and_dir() as (modelid, directory):
        p = run('training', modelid, directory, dataset)
        assert p.returncode == 0
        d = json.loads(p.stdout)
        assert d['status'] == 0


def test_evaluation():
    with temp_dataset('split-evaluation-train.bz2') as dataset, \
         temp_id_and_dir() as (modelid, directory):
        min_score = '0.6'
        accepted_deviation = '0.02'
        n_test_runs = '2'

        p = run('evaluation', modelid,
                directory,
                dataset,
                min_score,
                accepted_deviation,
                n_test_runs)

        assert p.returncode == 0

        d = json.loads(p.stdout)
        # let's make some *very* basic assertions about quality
        for k in ('balanced accuracy',
                  'accuracy',
                  'precision',
                  'recall',
                  'f1_score',
                  'score'):
            assert d[k] >= 0.7


def test_import_export():
    with temp_id_and_dir() as (import_id, import_dir):
        p = run('import', import_id, import_dir, TEST_MODEL)
        assert p.returncode == 0
        result = p.stdout.decode()
        assert result == 'Ok\nOk\n'

        assert os.path.isdir(import_dir)
        assert os.listdir(import_dir)  # it shouldn't be empty

        with temp_id_and_dir() as (export_id, export_dir):
            p = run('export', import_id, import_dir, export_dir)
            assert p.returncode == 0
            result = p.stdout.decode().strip()
            print(result)
            assert os.path.isdir(result)
            assert os.listdir(result)  # it shouldn't be empty
            assert result == export_dir


def _test_insuffient_args(module, n):
    modelid = unique_id()
    for i in range(n - 1):
        p = run(module, modelid, *['x' * i])
        result = json.loads(p.stdout)
        assert re.match('^\d+$', result['runid'])
        assert result['status'] == 1  # estimator.GENERAL_ERROR
        assert p.returncode == 1
        assert result['info'][0].startswith('Missing arguments')


def test_prediction_insuffient_args():
    _test_insuffient_args('prediction', 3)


def test_training_insuffient_args():
    _test_insuffient_args('training', 3)


def test_evaluation_insuffient_args():
    _test_insuffient_args('evaluation', 7)


def test_import_prediction():
    with temp_id_and_dir() as (import_id, import_dir):
        p = run('import', import_id, import_dir, TEST_MODEL)
        assert p.returncode == 0
        result = p.stdout.decode()
        assert result == 'Ok\nOk\n'

        assert os.path.isdir(import_dir)
        assert os.listdir(import_dir)  # it shouldn't be empty

        with temp_dataset('split-evaluation-predict.bz2') as dataset:
            p = run('prediction', import_id, import_dir, dataset)

        assert p.returncode == 0
        d = json.loads(p.stdout)
        assert d['status'] == 0

        # In test_webapp.py we do the same prediction tests using a
        # newly trained model (test_stashed_training_prediction); here
        # we do it with an import.

        answer_file = os.path.join(HERE,
                                   'test-requests',
                                   'split-evaluation-answers.bz2')

        with bz2.open(answer_file) as f:
            answers = json.load(f)

        predictions = d['predictions']
        assert isinstance(predictions, list)
        assert len(predictions) == len(answers)

        correct = 0
        for k, category, score in predictions:
            correct += int(category) == answers[k]
            assert isinstance(k, str)
            assert category in ('0', '1')
            assert 0.5 <= float(score) <= 1

        accuracy = correct / len(answers)
        assert accuracy > 0.8
        # baseline is how good you could get by always saying 1 or 0,
        # depending which is more common.
        baseline = sum(answers.values())
        baseline = max(baseline, len(answers) - baseline)
        assert correct > baseline
