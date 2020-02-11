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


