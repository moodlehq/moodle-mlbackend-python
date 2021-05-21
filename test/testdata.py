import random
from math import pi, cos, sin
import json
import csv

import numpy as np
from io import StringIO

class MoodleMLError(Exception):
    pass

MODEL_DTYPE='float32'


def discrete_dataset(n_inputs, n_outputs, n_samples, function):
    if n_outputs != 1:
        raise NotImplementedError("go ahead! implement n_outputs > 1!")
    n_classes = n_outputs + 1
    header = {
        'nfeatures': n_inputs,
        'nclasses': n_classes,
        'targettype': 'discrete',
        'targetclasses': '[0,1]'
    }

    colids = ['f%d' % i for i in range(n_inputs)]
    colids += ['t%d' % i for i in range(n_outputs)]

    dataset = function(n_inputs, n_outputs, n_samples)
    return (header, dataset, colids)


def cos_gt_sin(n_inputs, n_outputs, n_samples):
    assert n_inputs == 2 and n_outputs == 1
    rows = []
    for i in range(n_samples):
        a = random.uniform(-pi, pi)
        b = random.uniform(-pi, pi)
        c = float(cos(a) > sin(b))
        rows.append([a, b, c])
    return rows


def fake_dataset_cos_gt_sin(n_samples, train):

    data = discrete_dataset(2, 1,
                            n_samples=n_samples,
                            function=cos_gt_sin)

    if train:
        return bytesify_training(*data)
    else:
        return bytesify_prediction(*data)


def _prepare_headers(header):
    hk = []
    hv = []
    for k, v in header.items():
        hk.append(k)
        if isinstance(v, int):
            v = str(v)
        elif k == 'targetclasses':
            v = f'"{v}"'

        hv.append(v)
    return hk, hv

def bytesify_training(header, dataset, colids):
    hk, hv = _prepare_headers(header)
    out = [','.join(x) for x in (hk, hv, colids)]
    out.extend(','.join(str(x) for x in row) for row in dataset)
    return '\n'.join(out).encode('utf-8')


def bytesify_prediction(header, dataset, colids):
    # Here we return the query, then as a python list, the answers!
    hk, hv = _prepare_headers(header)
    colids = ['sampleid'] + colids
    out = [','.join(x) for x in (hk, hv, colids)]
    answers = []
    for i, row in enumerate(dataset):
        row = [f'r{i}'] + row
        answers.append(row.pop())
        out.append(','.join(str(x) for x in row))
    return ('\n'.join(out).encode('utf-8'), answers)
