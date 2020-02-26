"""Helpers to deal with stashed requests"""
import pickle
from pprint import pprint
import re
import csv
import random
import sys
import hashlib
import gzip
import bz2
import json

import numpy as np


def read_pickle(filename):
    # maybe the pickle is zipped.
    for o in (bz2.open,
              gzip.open,
              open):
        try:
            with o(filename, 'rb') as f:
                return pickle.load(f)
        except OSError as e:
            continue
    raise OSError(f"could not open '{filename}'")


def load(filename):
    a = read_pickle(filename)
    data = a['data']
    raw_headers = a['headers']
    url = a['url'].rsplit('/', 1)[1]
    kept_headers = {
        'Content-Type': 'content_type',
        'Content-Length': 'content_length',
    }
    headers = {}
    for k, v in raw_headers:
        if k in kept_headers:
            headers[kept_headers[k]] = v

    return data, headers, url


def get_boundary(headers):
    if isinstance(headers, dict):
        headers = headers.items()
    for k, v in headers:
        if k.lower() in ('content-type', 'content_type'):
            _, boundary = v.split('boundary=')
            return boundary.encode('utf8')


def get_uid(body, headers):
    return split_body(body, get_boundary(headers))['uniqueid'][1]

def split_body(body, boundary):
    """Split up a multipart form submission into a dictionary keyed by the
    content-disposition name.

    {
       name:  [headers, body],...
    }

    """
    # (If you thought there would be a standard library for this, you
    # wouldn't be the first).

    parts = {}
    for p in body.split(boundary):
        if len(p) < 5:
            continue
        headers, body = p.split(b'\r\n\r\n', 1)
        if body[-4:] == b'\r\n--':
            body = body[:-4]

        h2 = {}
        for h in headers.split(b'\r\n'):
            h = h.decode('utf8')
            try:
                k, v = h.split(':', 1)
            except ValueError:
                continue
            values = v.strip().split(';')
            h2[k] = {}
            for v in values:
                v = v.strip()
                if '=' in v:
                    a, b = v.split('=', 1)
                    h2[k][a] = b
                    if a == 'name' and k == 'Content-Disposition':
                        parts[b.replace('"', '')] = (h2, body)
                else:
                    h2[k][v] = None

    return parts


def reform_body(parts, boundary):
    out = [b'']
    for p in parts.values():
        headers, body = p
        hparts = []
        for hk, hv in headers.items():
            line = hk + ': '
            for k, v in hv.items():
                if v is None:
                    line += k
                else:
                    line += f'; {k}={v}'
            hparts.append(line)
        out.append('\r\n'.join(hparts).encode('utf8') +
                   b'\r\n\r\n' +
                   body +
                   b'\r\n--')

    s = (boundary + b'\r\n').join(out)
    return b'--' + s + boundary + b'--\r\n'


def fix_sample_id(lines, sid):
    # we don't bother with full csv parsing, because these are all numbers
    # except the id
    for i, line in enumerate(lines):
        if ',' in line:
            bits = line.split(',')
            bits[sid] = f'sample-{i + 1}'
            lines[i] = ','.join(bits)


def shuffle_values(lines, sampleid, targetcolumn, shuffle_cols=True, seed=None):
    # We put the lines in a different order, otherwise unchanged.
    if seed is not None:
        random.seed(seed)
        npseed = np.frombuffer(hashlib.sha512(seed).digest(), dtype='uint32')
        np.random.seed(npseed)
    random.shuffle(lines)
    if not shuffle_cols:
        return
    # now we put the columns in a different order, but not the
    # sampleid or target rows, which we vant to retain.
    input_rows = []
    sampleids = []
    targets = []
    for line in lines:
        vals = line.split(',')
        if sampleid is not None:
            sampleids.append(vals.pop(sampleid))
        if targetcolumn is not None:
            targets.append(vals.pop())
        input_rows.append(vals)

    a = np.asarray(input_rows)
    np.random.shuffle(a.T)
    input_rows = a.tolist()

    if sampleids:
        for s, row in zip(sampleids, input_rows):
            row.insert(sampleid, s)

    if targets:
        for t, row in zip(targets, input_rows):
            row.append(t)

    for i, row in enumerate(input_rows):
        lines[i] = ','.join(row)


def anonymize_dataheader(h):
    # filename can contain date information
    if 'Content-Disposition' in h:
        h['Content-Disposition']['filename'] = 'data.csv'


def anonymize_dataset(dataset, contains_sampleid=False, seed=None):
    lines = dataset.decode('utf8').strip().split('\n')

    header = lines[:3]
    del lines[:3]

    h = csv.reader(header)
    metakeys = next(h)
    metavals = next(h)

    targetcolumn = None
    newkeys = []
    newvals = []

    for k, v in zip(metakeys, metavals):
        if k == 'targetcolumn':
            newkeys.append(k)
            newvals.append('TARGET')
            targetcolumn = v
        elif k in ('nfeatures', 'targettype'):
            newkeys.append(k)
            newvals.append(v)
        elif k in ('targetclasses',):
            newkeys.append(k)
            newvals.append(f'"{v}"')

    cols = next(h)

    sampleid = None
    newcols = []
    for i, col in enumerate(cols):
        if col == targetcolumn:
            newcols.append('TARGET')
        else:
            newcols.append(f'input-{i}')
        if col == 'sampleid':
            sampleid = i

    header[:] = [
        ','.join(newkeys),
        ','.join(newvals),
        ','.join(newcols)
    ]

    if sampleid not in (None, 0):
        raise ValueError("sampleid column is not zero")
    if targetcolumn and targetcolumn in cols[:-1]:
        raise ValueError("target is not last column")

    shuffle_values(lines,
                   sampleid,
                   targetcolumn,
                   shuffle_cols=True,
                   seed=seed)

    if sampleid is not None:
        if not contains_sampleid:
            raise ValueError(f"a sampleid column found but unexpected!")
        fix_sample_id(lines, sampleid)
    elif contains_sampleid:
        raise ValueError(f"a sampleid column was expected but not found!")

    return '\n'.join(header + lines).encode('utf8')


def anonymize(filename, seed=None):
    a = read_pickle(filename)
    raw_data = a['data']
    raw_headers = a['headers']
    raw_url  = a['url']

    url = re.sub(r'^http[s]?://[^/]+',
                 'http://example.com',
                 raw_url)

    headers = []

    good_headers = {
        'content-type',
    }

    for h in raw_headers:
        if h[0].lower() in good_headers:
            headers.append(h)

    boundary = get_boundary(headers)

    parts = split_body(raw_data, boundary)

    data_header, dataset = parts['dataset']
    is_predict = 'prediction' in url

    if seed is None:
        seed = parts['uniqueid'][1]

    dataset = anonymize_dataset(dataset, is_predict, seed=seed)
    anonymize_dataheader(data_header)
    parts['uniqueid'] = rehash(parts['uniqueid'], seed)
    parts['dirhash'] = rehash(parts['dirhash'], seed)
    parts['dataset'] = (data_header, dataset)

    data = reform_body(parts, boundary)
    headers.append(('Content-length', str(len(data))))

    r = {
        'url':   url,
        'data':  data,
        'headers': headers,
    }

    return pickle.dumps(r)


def rehash(x, seed):
    h, s = x
    size = max(20, len(s))
    hash = hashlib.sha256(s)
    hash.update(seed)
    return (h, hash.hexdigest().encode('utf-8')[:size])
