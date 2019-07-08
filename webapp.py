import os
import re
import json
import tempfile
import zipfile
import shutil

from flask import Flask, request, send_file, Response
from werkzeug.utils import secure_filename

from moodlemlbackend.processor import estimator

app = Flask(__name__)


def get_request_value(key, pattern=False, exception=True):

    if pattern is False:
        pattern = '[^A-Za-z0-9_\-$]'

    value = request.values.get(key)
    if value is None:

        if exception is True:
            raise Exception('The requested key ' + key + ' is not available.')
        return False

    return re.sub(pattern, '', value)


def get_model_dir(hashkey=False):

    basedir = os.environ["MOODLE_MLBACKEND_PYTHON_DIR"]

    if os.path.exists(basedir) is False:
        raise IOError(
            'The base dir does not exist. ' +
            'Set env MOODLE_MLBACKEND_PYTHON_DIR to an existing dir')

    os.access(basedir, os.W_OK)

    uniquemodelid = get_request_value('uniqueid')

    # The dir in the server is namespaced by uniquemodelid and the
    # dirhash (if present) which determines where the results should be stored.
    modeldir = os.path.join(basedir, uniquemodelid)

    if hashkey is not False:
        dirhash = get_request_value(hashkey)
        modeldir = os.path.join(modeldir, dirhash)

    return modeldir


def check_access():

    envvarname = "MOODLE_MLBACKEND_PYTHON_USERS"
    if envvarname not in os.environ:
        raise Exception(
            envvarname + ' environment var is not set in the server.')

    if re.match(os.environ[envvarname], '[^A-Za-z0-9_\-,$]'):
        raise Exception(
            'The value of ' + envvarname + ' environment var does not ' +
            ' adhere to [^A-Za-z0-9_\-,$]')

    users = os.environ[envvarname].split(',')

    if (request.authorization is None or
            request.authorization.username is None or
            request.authorization.password is None):
        return 'No user and/or password included in the request.'

    for user in users:
        userdata = user.split(':')
        if len(userdata) != 2:
            raise Exception('Incorrect format for ' +
                            envvarname + ' environment var. It should ' +
                            'contain a comma-separated list of ' +
                            'username:password.')

        if (userdata[0] == request.authorization.username and
                userdata[1] == request.authorization.password):
            return True

    return 'Incorrect user and/or password provided by Moodle.'


def get_file_path(filekey):

    file = request.files[filekey]

    # We can use a temp directory for the input files.
    tempdir = tempfile.mkdtemp()
    filepath = os.path.join(tempdir, secure_filename(file.filename))
    file.save(filepath)

    return filepath


def zipdir(dirpath, zipfilepath):

    ziph = zipfile.ZipFile(zipfilepath, 'w', zipfile.ZIP_DEFLATED)

    for root, dirs, files in os.walk(dirpath):
        for file in files:
            abspath = os.path.join(root, file)
            ziph.write(abspath, os.path.relpath(abspath, root))
    ziph.close()
    return ziph


@app.route('/version', methods=['GET'])
def version():
    here = os.path.abspath(os.path.dirname(__file__))
    version_file = open(os.path.join(here, 'moodlemlbackend', 'VERSION'))
    return version_file.read().strip()


@app.route('/training', methods=['POST'])
def training():

    access = check_access()
    if access is not True:
        return access, 401

    uniquemodelid = get_request_value('uniqueid')
    outputdir = get_model_dir('outputdirhash')

    datasetpath = get_file_path('dataset')

    classifier = estimator.Classifier(uniquemodelid, outputdir)
    result = classifier.train_dataset(datasetpath)

    return json.dumps(result)


@app.route('/prediction', methods=['POST'])
def prediction():

    access = check_access()
    if access is not True:
        return access, 401

    uniquemodelid = get_request_value('uniqueid')
    outputdir = get_model_dir('outputdirhash')

    datasetpath = get_file_path('dataset')

    classifier = estimator.Classifier(uniquemodelid, outputdir)
    result = classifier.predict_dataset(datasetpath)

    return json.dumps(result)


@app.route('/evaluation', methods=['POST'])
def evaluation():

    access = check_access()
    if access is not True:
        return access, 401

    uniquemodelid = get_request_value('uniqueid')
    outputdir = get_model_dir('outputdirhash')

    minscore = get_request_value('minscore', pattern='[^0-9.$]')
    maxdeviation = get_request_value('maxdeviation', pattern='[^0-9.$]')
    niterations = get_request_value('niterations', pattern='[^0-9$]')

    datasetpath = get_file_path('dataset')

    trainedmodeldirhash = get_request_value(
        'trainedmodeldirhash', exception=False)
    if trainedmodeldirhash is not False:
        # The trained model dir in the server is namespaced by uniquemodelid
        # and the trainedmodeldirhash which determines where should the results
        # be stored.
        trainedmodeldir = get_model_dir('trainedmodeldirhash')
    else:
        trainedmodeldir = False

    classifier = estimator.Classifier(uniquemodelid, outputdir)
    result = classifier.evaluate_dataset(datasetpath,
                                                float(minscore),
                                                float(maxdeviation),
                                                int(niterations),
                                                trainedmodeldir)

    return json.dumps(result)


@app.route('/evaluationlog', methods=['GET'])
def evaluationlog():

    access = check_access()
    if access is not True:
        return access, 401

    outputdir = get_model_dir('outputdirhash')
    runid = get_request_value('runid', '[^0-9$]')
    logsdir = os.path.join(outputdir, 'logs', runid)

    zipfile = tempfile.NamedTemporaryFile()
    zipdir(logsdir, zipfile)
    return send_file(zipfile.name, mimetype='application/zip')


@app.route('/export', methods=['GET'])
def export():

    access = check_access()
    if access is not True:
        return access, 401

    uniquemodelid = get_request_value('uniqueid')
    modeldir = get_model_dir('modeldirhash')

    # We can use a temp directory for the export data
    # as we don't need to keep it forever.
    tempdir = tempfile.mkdtemp()

    classifier = estimator.Classifier(uniquemodelid, modeldir)
    exportdir = classifier.export_classifier(tempdir)
    if exportdir is False:
        return Response('There is nothing to export.', 503)

    zipfile = tempfile.NamedTemporaryFile()
    zipdir(exportdir, zipfile)
    return send_file(zipfile.name, mimetype='application/zip')


@app.route('/import', methods=['POST'])
def import_model():

    access = check_access()
    if access is not True:
        return access, 401

    uniquemodelid = get_request_value('uniqueid', '')
    modeldir = get_model_dir('modeldirhash')

    importzippath = get_file_path('importzip')

    with zipfile.ZipFile(importzippath, 'r') as zipobject:
        importtempdir = tempfile.mkdtemp()
        zipobject.extractall(importtempdir)

        classifier = estimator.Classifier(uniquemodelid, modeldir)
        classifier.import_classifier(importtempdir)

    return 'Ok', 200


@app.route('/deletemodel', methods=['POST'])
def deletemodel():

    access = check_access()
    if access is not True:
        return access, 401

    modeldir = get_model_dir()

    if os.path.exists(modeldir):
        # The directory may not exist.
        shutil.rmtree(modeldir, False)

    return 'Ok', 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
