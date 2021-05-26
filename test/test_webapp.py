import os
import re
from base64 import b64encode
from io import BytesIO
import secrets
from zipfile import ZipFile
import json
from contextlib import contextmanager
from pprint import pprint
import time
import random
import inspect
import subprocess
import numpy as np

import stash
import pytest
from flask import url_for
import testdata
import bz2

# Set USE_ENV_USERS to true to read users and passwords from the
# environment rather than a file.
USE_ENV_USERS = True

HERE = os.path.dirname(__file__)
USERS = {
    'a': 'b',
    'kaka': 'kea',
}


def gen_password_string():
    # set this environment var to test the old style
    if os.environ.get("MOODLE_MLBACKEND_OLD_PASSWORDS"):
        return ','.join(':'.join(x) for x in USERS.items())

    exe = os.path.join(HERE, '..', 'gen-passwd')
    cmd = [exe]
    cmd.extend(USERS.keys())
    cmd.append('-P')
    cmd.append(','.join(USERS.values()))
    p = subprocess.run(cmd,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       check=True)

    return p.stdout.decode('utf8').strip()


# pylint et.al. will GNASH THEIR TEETH at this, but we need to set up
# the environment before importing webapp.

DATA_DIR = os.path.join(HERE, 'temp-data')
TEMP_DIR = os.path.join(HERE, 'temp-data/tmp')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["MOODLE_MLBACKEND_PYTHON_DIR"] = DATA_DIR
os.environ["MOODLE_MLBACKEND_TEMPDIR"] = TEMP_DIR

os.environ["MOODLE_MLBACKEND_PYTHON_USERS"] = gen_password_string()

import webapp

# env setting to prevent deletion of (some) test models, for post
# mortum diagnostics.
KEEP_TEST_MODELS = os.environ.get("MOODLE_MLBACKEND_KEEP_TEST_MODELS")

# env setting to run normally skipped tests
RUN_SLOW_TESTS = os.environ.get("MOODLE_MLBACKEND_RUN_SLOW_TESTS")


BAD_UID = 'an unused unique ID that does not exist ' + secrets.token_hex(10)


@pytest.fixture
def app():
    app = webapp.app
    app.config['DEBUG'] = True
    app.config['TESTING'] = True
    return app


def _auth(user=None, password=None):
    if user is None:
        return {}

    if password is None:
        password = USERS[user]

    s = f"{user}:{password}"
    auth = b64encode(s.encode('utf-8'))
    return {"Authorization": b"Basic " + auth}


def get_dataset(n=100, train=True):
    d = testdata.fake_dataset_cos_gt_sin(n, train)
    if isinstance(d, tuple):
        x, y = d
        return ((BytesIO(x), 'x.txt'), y)

    return (BytesIO(d), 'data.txt')


def _import_post(post, uniqueid='1', zipdata=None, **kwargs):

    if zipdata is None:
        zipdata = _get_import_zip()

    data = {
        'uniqueid': uniqueid,
        'dirhash': 'abc',
        'importzip': zipdata,
    }
    return post(url_for('import_model'),
                content_type='multipart/form-data',
                data=data,
                **kwargs)


def to_zipfile(x):
    #DWTFIM for bytes or ZipFile objects.
    if isinstance(x, ZipFile):
        return x
    if isinstance(x, bytes):
        x = BytesIO(x)
    return ZipFile(x)


def _cmp_zipdata(a, b):
    # We can't directly compare zip data because the files inside have
    # timestamps, which in our case are meaningless but inconsistent.
    a = to_zipfile(a)
    b = to_zipfile(b)
    ainfo = a.infolist()
    anames = a.namelist()
    binfo = b.infolist()
    bnames = b.namelist()
    # infolist can be longer than namelist if there are duplicate names.
    # we can call that an error then deal with names which is easier.
    assert len(ainfo) == len(anames)
    assert len(binfo) == len(bnames)
    assert len(anames) == len(bnames)
    # we have the same names:
    assert set(anames + bnames) == set(anames)

    for name in anames:
        with a.open(name) as af, b.open(name) as bf:
            acontent = af.read()
            bcontent = bf.read()
            assert acontent == bcontent


def _training_post(post, uniqueid='1', dataset=None, **kwargs):
    """Submit a dataset for training.

    - post is usually client.post
    - kwargs should include headers=auth.
    """
    if dataset is None:
        dataset = get_dataset(train=True)
    data = {
        'uniqueid': uniqueid,
        'dirhash': 'abc',
        'dataset': dataset,
    }

    return post(url_for('training'), data=data, **kwargs)


def _prediction_post(client, uniqueid='1', n=100, x=None, y=None, **kwargs):
    """Submit a dataset for prediction. The answer is json.
    The dataset should lack target values.

    - post is usually client.post
    - kwargs should include headers=auth.
    """
    if x is None:
        x, y = get_dataset(n=n, train=False)
    data = {
        'uniqueid': uniqueid,
        'dirhash': 'abc',
        'dataset': x
    }
    return client.post(url_for('prediction'), data=data, **kwargs), y


def _evaluation_post(client, uniqueid, dataset=None, n=10000, **kwargs):
    """Submit a dataset for evaluation"""
    if dataset is None:
        dataset = get_dataset(n=n)

    data = {
        'uniqueid': uniqueid,
        'dirhash': 'abc',
        'minscore': '0',
        'maxdeviation': '0',
        'niterations': '1',
        'dataset': dataset
    }
    return client.post(url_for('evaluation'), data=data, **kwargs)


def _evaluationlog_get(get, uniqueid=None, **kwargs):
    # a helper that does all the right things except auth
    if uniqueid is None:
        uniqueid = BAD_UID

    data = {
        'uniqueid': uniqueid,
        'dirhash': 'abc',
        'runid': 0,
    }
    return get(url_for('evaluationlog'), data=data, **kwargs)


def _random_weights(shape, mean=0.0, sd=0.1):
    return np.random.normal(mean, sd, shape).tolist()


def _get_import_zip(n_hidden=10,
                    add_random_weights=True):
    # we need to double the outputs because that is what the backend does.
    n_classes, n_features = 2, 2
    d = {
        'n_features': n_features,
        'n_classes': n_classes,
        'n_hidden': n_hidden,
    }
    if add_random_weights:
        d['initialise-vars/input-to-hidden-weights'] = _random_weights((n_features, n_hidden))
        d['initialise-vars/hidden-to-output-weights'] = _random_weights((n_hidden, n_classes))
        d['initialise-vars/hidden-bias'] = _random_weights(n_hidden)
        d['initialise-vars/output-bias'] = _random_weights(n_classes)

    b = BytesIO()
    zf = ZipFile(b, mode='w')

    zf.writestr('model.json', json.dumps(d))
    zf.close()
    return (BytesIO(b.getvalue()), 'model.zip')


def _delete_model(post, uniqueid='1', **kwargs):
    data = {
        'uniqueid': uniqueid,
    }
    return post(url_for('deletemodel'), data=data, **kwargs)


@contextmanager
def temporary_model(client, uniqueid=None, auth=None, **kwargs):
    """Context manager to automatically delete a model after use.
    It yeilds a unique ID and auth header. Like this:

    with temporary_model(client) as (uid, auth):
        resp = _training_post(client.post,
                              uniqueid=uid,
                              headers=auth)
    """
    if auth is None:
        auth = _auth('a')
    if uniqueid is None:
        name = inspect.currentframe().f_back.f_back.f_code.co_name
        uniqueid = f'temp model for {name} at {time.asctime()} ({time.time()})'
    try:
        #r = _import_post(client.post, uniqueid, headers=auth, **kwargs)
        #assert r.status_code == 200
        yield (uniqueid, auth)
    finally:
        _delete_model(client.post, uniqueid, headers=auth)


def post_real_data_no_cleanup(post, filename, url=None, **kwargs):
    if kwargs == {}:
        kwargs = _auth('a')

    data, content_headers, _url = stash.load(filename)

    if url is None:
        url = url_for(_url)
    return post(url, data=data, headers=kwargs, **content_headers)


@contextmanager
def post_real_data(post, filename, url=None, extra_args=None, **kwargs):
    """Replay a stashed request.

    - post is usually client.post
    - kwargs are headers; if empty, correct auth is used.
    """
    if kwargs == {}:
        kwargs = _auth('a')

    data, content_headers, _url = stash.load(filename)
    uid = stash.get_uid(data, content_headers)

    if extra_args is not None:
        data = stash.set_args(data, content_headers, extra_args)

    if url is None:
        url = url_for(_url)

    try:
        yield post(url, data=data, headers=kwargs, **content_headers)
    finally:
        if not KEEP_TEST_MODELS:
            _delete_model(post, uniqueid=uid, headers=kwargs)


def _export_get(get, uniqueid='1', **kwargs):
    # a helper that does all the right things except auth
    data = {
        'uniqueid': uniqueid,
        'dirhash': '123'
    }
    return get(url_for('export'), data=data, **kwargs)


def test_server_tempfile(client): # NOTE: not using client.
    with webapp.tempfile.TemporaryDirectory() as d:
        assert os.path.dirname(d) == TEMP_DIR
        assert os.path.isdir(d)
    assert not os.path.exists(d)


def test_version(client):
    resp = client.get(url_for('version'))
    assert resp.status_code == 200
    assert re.match(br'^\d+\.\d+\.\d+\n?', resp.data) is not None


def test_version_post(client):
    resp = client.post(url_for('version'))
    assert resp.status_code == 405


def test_version_with_auth(client):
    # /version should work with or without authentication
    auth = _auth('a')
    resp = client.get(url_for('version'), headers=auth)
    assert resp.status_code == 200
    assert re.match(br'^\d+\.\d+\.\d+\n?', resp.data) is not None


def test_training_no_auth(client):
    resp = _training_post(client.post)
    assert resp.status_code == 401


def test_training_no_auth_get(client):
    resp = _training_post(client.get)
    assert resp.status_code == 405


def test_training_bad_auth(client):
    bad_auth = _auth('x', 'y')
    resp = _training_post(client.post, headers=bad_auth)
    assert resp.status_code == 401


def test_training_only(client):
    with temporary_model(client) as (uid, auth):
        resp = _training_post(client.post,
                              uniqueid=uid,
                              headers=auth)
        assert resp.status_code == 200
        results = json.loads(resp.data)


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="quite long")
def test_stashed_training_short(client):
    filename = os.path.join(HERE,
                            'test-requests',
                            'test-366-1904-training.bz2')

    with post_real_data(client.post, filename) as resp:
        assert resp.status_code == 200
        results = json.loads(resp.data)
        pprint(results)


def _stashed_evaluation(client,
                        filename,
                        expected_ranges,
                        niterations=3):
    url = url_for('evaluation')
    # the backend wants to make assertions about the expected score
    # and deviation, but we do that here instead.
    args = {'minscore': 0,
            'maxdeviation': 1.0,
            'niterations': niterations
            }

    with post_real_data(client.post,
                        filename,
                        url=url,
                        extra_args=args) as resp:
        assert resp.status_code == 200
        results = json.loads(resp.data)
        pprint(results)

    for k, v in expected_ranges.items():
        r = results[k]
        low, high = v
        assert low <= r <= high

    return results


def test_stashed_evaluation_short(client):
    filename = os.path.join(HERE,
                            'test-requests',
                            'test-366-1904-training.bz2'
    )
    expected_ranges = {
        'accuracy': [0.75, 0.95],
        'f1_score': [0.75, 0.95],
        'precision': [0.70, 0.95],
        'recall': [0.70, 0.95],
        'score': [0.75, 0.95],
        'score_deviation': [0.0, 0.0],
        'status': [0, 0],
        'min_score': [0.0, 0.0],
        'accepted_deviation': [1.0, 1.0],
        #'auc' and 'auc_deviation' are broken
    }

    _stashed_evaluation(client,
                        filename,
                        expected_ranges,
                        niterations=1)


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow")
def test_stashed_evaluation_long(client):
    filename = os.path.join(HERE,
                            'test-requests',
                            'test-415-37953-training.bz2'
    )

    expected_ranges = {
        'accuracy': [0.75, 0.95],
        'f1_score': [0.75, 0.95],
        'precision': [0.75, 0.95],
        'recall': [0.75, 0.95],
        'score': [0.75, 0.95],
        'score_deviation': [0.0, 0.05],
        'status': [0, 0],
        'min_score': [0.0, 0.0],
        'accepted_deviation': [1.0, 1.0],
        #'auc' and 'auc_deviation' are broken
    }

    _stashed_evaluation(client,
                        filename,
                        expected_ranges,
                        niterations=3)


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="long")
def test_stashed_training_long(client):
    filename = os.path.join(HERE,
                            'test-requests',
                            'test-415-37953-training.bz2'
    )
    with post_real_data(client.post, filename) as resp:
        assert resp.status_code == 200
        results = json.loads(resp.data)
        pprint(results)


def test_stashed_training_prediction(client):
    train = os.path.join(HERE,
                         'test-requests',
                         'split-evaluation-train.bz2'
    )
    predict = os.path.join(HERE,
                           'test-requests',
                           'split-evaluation-predict.bz2'
    )

    answer_file = os.path.join(HERE,
                               'test-requests',
                               'split-evaluation-answers.bz2'
    )

    with bz2.open(answer_file) as f:
        answers = json.load(f)

    with post_real_data(client.post, train) as resp:
        assert resp.status_code == 200
        results = json.loads(resp.data)
        pprint(results)
        resp_p = post_real_data_no_cleanup(client.post, predict)

        assert resp_p.status_code == 200
        results = json.loads(resp_p.data)
        assert 'predictions' in results
        predictions = results['predictions']
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

        print(f"correct: {correct}/{len(answers)} == {accuracy}")


def test_stashed_double_train(client):
    train1 = os.path.join(HERE,
                         'test-requests',
                         'double-train-train.bz2'
    )
    train2 = os.path.join(HERE,
                         'test-requests',
                         'double-train-train2.bz2'
    )
    predict = os.path.join(HERE,
                           'test-requests',
                           'double-train-predict.bz2'
    )

    answer_file = os.path.join(HERE,
                               'test-requests',
                               'double-train-answers.bz2'
    )

    with bz2.open(answer_file) as f:
        answers = json.load(f)

    with post_real_data(client.post, train1) as resp:
        assert resp.status_code == 200
        results = json.loads(resp.data)
        pprint(results)
        resp_t2 = post_real_data_no_cleanup(client.post, train2)
        assert resp_t2.status_code == 200
        results = json.loads(resp_t2.data)
        pprint(results)

        resp_p = post_real_data_no_cleanup(client.post, predict)

        assert resp_p.status_code == 200
        results = json.loads(resp_p.data)
        assert 'predictions' in results
        predictions = results['predictions']
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

        print(f"correct: {correct}/{len(answers)} == {accuracy}")


def test_training_prediction_evaluation(client):
    zipdata = _get_import_zip(n_hidden=10)

    dataset = get_dataset(n=10000)

    with temporary_model(client, zipdata=zipdata) as (uid, auth):
        resp = _training_post(client.post,
                              uniqueid=uid,
                              dataset=dataset,
                              headers=auth)

        assert resp.status_code == 200
        results = json.loads(resp.data)

        resp, expected = _prediction_post(client,
                                          uniqueid=uid,
                                          n=200,
                                          headers=auth)
        data = json.loads(resp.data)
        results = [float(x[1]) for x in data['predictions']]

        assert len(results) == len(expected)
        correct = [a == b for a, b in zip(results, expected)]
        accuracy = sum(correct) / len(correct)
        assert accuracy > 0.8

        eval_data = get_dataset(n=10000)

        resp = _evaluation_post(client,
                                uid,
                                dataset=eval_data,
                                headers=auth)

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data['accuracy'] > 0.8
        # because this is a balanced artificial dataset, we can be
        # fairly sure the precision and recall follow the accuracy.
        assert data['recall'] > 0.8
        assert data['precision'] > 0.8
        assert data['score'] > 0.8
        assert data['f1_score'] > 0.8


def test_prediction_bad_auth(client):
    bad_auth = _auth('x', 'y')
    resp, y = _prediction_post(client, headers=bad_auth)
    assert resp.status_code == 401


def test_prediction_no_model(client):
    auth = _auth('a')
    resp, y = _prediction_post(client, headers=auth, uniqueid='non-existent')
    # TODO: a 4xx response seems better
    assert resp.status_code == 200
    result = json.loads(resp.data)
    assert result['status'] == 2
    assert result['info'] == ['Provided model have not been trained yet']


def test_prediction_untrained(client):
    with temporary_model(client) as (uid, auth):
        n = 100
        resp, y = _prediction_post(client, uniqueid=uid, n=n, headers=auth)
        data = json.loads(resp.data)
        assert data == {
            'status': 2,
            'info': ['Provided model have not been trained yet']
        }
        # XXX 200 is really a lie
        assert resp.status_code == 200


def test_evaluation_bad_auth(client):
    bad_auth = _auth('x', 'y')
    with temporary_model(client) as (uid, auth):
        resp = _evaluation_post(client, uid, headers=bad_auth, n=100)
        assert resp.status_code == 401


def test_evaluation(client):
    with temporary_model(client) as (uid, auth):
        resp = _evaluation_post(client, uid, headers=auth)
        assert resp.status_code == 200
        results = json.loads(resp.data)

        # We just want to know it learnt *something*
        for indicator, low, high in (#('auc', 0, 1),       # AUC is broken
                                     ('accuracy', 0.7, 1),
                                     ('precision', 0.7, 1),
                                     ('recall', 0.7, 1),
                                     ('f1_score', 0.7, 1),
                                     ('score', 0.7, 1)):
            score = results[indicator]
            assert score >= low
            assert score <= high
        log_resp = _evaluationlog_get(client.get,
                                      uniqueid=uid,
                                      headers=auth)
        # log_resp payload is an empty zipfile
        # should it be?


def test_evaluationlog_bad_auth(client):
    bad_auth = _auth('x', 'y')
    resp = _evaluationlog_get(client.get, headers=bad_auth)
    assert resp.status_code == 401


def test_evaluationlog_no_model(client):
    auth = _auth('kaka')
    resp = _evaluationlog_get(client.get,
                              headers=auth,
    )
    assert resp.status_code == 200


def test_evaluationlog_post(client):
    auth = _auth('kaka')
    resp = _evaluationlog_get(client.post, headers=auth)
    assert resp.status_code == 405


def test_export_bad_auth(client):
    bad_auth = _auth('a', 'y')
    resp = _export_get(client.get, headers=bad_auth)
    assert resp.status_code == 401


def test_export_bad_uid(client):
    auth = _auth('kaka')
    resp = _export_get(client.get,
                       uniqueid=BAD_UID,
                       headers=auth)
    # TODO: A 4xx response would make more sense
    # but this is what we actually get.
    assert resp.status_code == 503


def test_import_bad_auth(client):
    bad_auth = _auth('a', 'y')
    resp = _import_post(client.post, headers=bad_auth)
    assert resp.status_code == 401
    bad_auth = _auth('as', 'y')
    resp = _import_post(client.post, headers=bad_auth)
    assert resp.status_code == 401


@pytest.mark.xfail
def test_double_import(client):
    auth = _auth('kaka')
    try:
        resp = _import_post(client.post, headers=auth)
        assert resp.status_code == 200
        resp = _import_post(client.post, headers=auth)
        assert resp.status_code == 400
    finally:
        _delete_model(client.post, headers=auth)
        # once it is deleted, we can import it again
        resp = _import_post(client.post, headers=auth)
        assert resp.status_code == 200
        _delete_model(client.post, headers=auth)
