import re
import os
import zipfile
import tempfile
import shutil
import atexit

from flask import request


def get_request_value(key, pattern=False, exception=True):

    if pattern is False:
        pattern = r'[^A-Za-z0-9_\-$]'

    value = request.values.get(key)
    if value is None:

        if exception is True:
            raise Exception('The requested key ' + key + ' is not available.')
        return False

    return re.sub(pattern, '', value)


def get_file_path(localbasedir, filekey):

    file = request.files[filekey]

    tempdir = tempfile.mkdtemp()
    tempfilepath = os.path.join(tempdir, filekey)

    atexit.register(shutil.rmtree, tempdir)
    file.save(tempfilepath)

    return tempfilepath


def zipdir(dirpath, zipf):

    ziph = zipfile.ZipFile(zipf, 'w', zipfile.ZIP_DEFLATED)

    for root, dirs, files in os.walk(dirpath):
        for file in files:
            abspath = os.path.join(root, file)
            ziph.write(abspath, os.path.relpath(abspath, root))
    ziph.close()
    return ziph
