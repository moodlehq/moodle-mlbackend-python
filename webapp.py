import os
import json
import tempfile
import zipfile

from flask import Flask, send_file, Response

from moodlemlbackend.processor import estimator
from moodlemlbackend.webapp.access import check_access
from moodlemlbackend.webapp.util import get_request_value, get_file_path
from moodlemlbackend.webapp.util import zipdir

app = Flask(__name__)

# If the MOODLE_MLBACKEND_PYTHON_S3_BUCKET_NAME environment variable is set, we
# use the S3 backend, otherwise the local file system.
#
# Note that there are conditional imports here, and linters might
# complain that they are not right at the top. Those linters should
# relax. What we are doing here is completely fine.

if "MOODLE_MLBACKEND_PYTHON_S3_BUCKET_NAME" in os.environ:
    from moodlemlbackend.webapp.s3 import S3, S3_setup_base_dir
    storage = S3()
    setup_base_dir = S3_setup_base_dir
else:
    from moodlemlbackend.webapp.localfs import LocalFS, LocalFS_setup_base_dir
    storage = LocalFS()
    setup_base_dir = LocalFS_setup_base_dir


# Set MOODLE_MLBACKEND_TEMPDIR to use something other than /tmp for
# temporary storage.
if "MOODLE_MLBACKEND_TEMPDIR" in os.environ:
    tempfile.tempdir = os.environ['MOODLE_MLBACKEND_TEMPDIR']


@app.route('/version', methods=['GET'])
def version():
    here = os.path.abspath(os.path.dirname(__file__))
    # "./web-compat-version" is NOT in the git tree.
    # If it exists, the version there is used; otherwise the true
    # version is given. This allows versions with interchangeable APIs
    # to be substituted for each other, allowing the ML-backend to be
    # upgraded at a different cadence than Moodle itself.
    # (Moodle makes a strict version check).
    for fn in [os.path.join(here, 'web-compat-version'),
               os.path.join(here, 'moodlemlbackend', 'VERSION')]:
        if os.path.exists(fn):
            with open(fn) as f:
                return f.read().strip()


@app.route('/training', methods=['POST'])
@check_access
@setup_base_dir(storage, True, True)
def training():

    uniquemodelid = get_request_value('uniqueid')
    modeldir = storage.get_model_dir('dirhash')

    with get_file_path(storage.get_localbasedir(), 'dataset') as dataset:
        classifier = estimator.Classifier(uniquemodelid, modeldir, dataset=dataset)
        result = classifier.train_dataset(dataset)

    return json.dumps(result)


@app.route('/prediction', methods=['POST'])
@check_access
@setup_base_dir(storage, True, True)
def prediction():

    uniquemodelid = get_request_value('uniqueid')
    modeldir = storage.get_model_dir('dirhash')

    with get_file_path(storage.get_localbasedir(), 'dataset') as datasetpath:
        classifier = estimator.Classifier(uniquemodelid, modeldir, datasetpath)
        result = classifier.predict_dataset(datasetpath)

    return json.dumps(result)


@app.route('/evaluation', methods=['POST'])
@check_access
@setup_base_dir(storage, False, False)
def evaluation():

    uniquemodelid = get_request_value('uniqueid')
    modeldir = storage.get_model_dir('dirhash')

    minscore = get_request_value('minscore', pattern='[^0-9.$]')
    maxdeviation = get_request_value('maxdeviation', pattern='[^0-9.$]')
    niterations = get_request_value('niterations', pattern='[^0-9$]')

    trainedmodeldirhash = get_request_value(
        'trainedmodeldirhash', exception=False)
    if trainedmodeldirhash is not False:
        # The trained model dir in the server is namespaced by uniquemodelid
        # and the trainedmodeldirhash which determines where should the results
        # be stored.
        trainedmodeldir = storage.get_model_dir(
            'trainedmodeldirhash', fetch_model=True)
    else:
        trainedmodeldir = False

    with get_file_path(storage.get_localbasedir(), 'dataset') as datasetpath:
        classifier = estimator.Classifier(uniquemodelid, modeldir, datasetpath)
        result = classifier.evaluate_dataset(datasetpath,
                                             float(minscore),
                                             float(maxdeviation),
                                             int(niterations),
                                             trainedmodeldir)

    return json.dumps(result)


@app.route('/evaluationlog', methods=['GET'])
@check_access
@setup_base_dir(storage, True, False)
def evaluationlog():

    modeldir = storage.get_model_dir('dirhash')
    runid = get_request_value('runid', '[^0-9$]')
    logsdir = os.path.join(modeldir, 'logs', runid)

    zipf = tempfile.NamedTemporaryFile()
    zipdir(logsdir, zipf)
    return send_file(zipf.name, mimetype='application/zip')


@app.route('/export', methods=['GET'])
@check_access
@setup_base_dir(storage, True, False)
def export():

    uniquemodelid = get_request_value('uniqueid')
    modeldir = storage.get_model_dir('dirhash')

    # We can use a temp directory for the export data
    # as we don't need to keep it forever.
    tempdir = tempfile.TemporaryDirectory()

    classifier = estimator.Classifier(uniquemodelid, modeldir)
    exportdir = classifier.export_classifier(tempdir.name)
    if exportdir is False:
        return Response('There is nothing to export.', 503)

    zipf = tempfile.NamedTemporaryFile()
    zipdir(exportdir, zipf)

    return send_file(zipf.name, mimetype='application/zip')


@app.route('/import', methods=['POST'])
@check_access
@setup_base_dir(storage, False, True)
def import_model():

    uniquemodelid = get_request_value('uniqueid')
    modeldir = storage.get_model_dir('dirhash')

    with get_file_path(storage.get_localbasedir(), 'importzip') as importzippath:
        with zipfile.ZipFile(importzippath, 'r') as zipobject:
            with tempfile.TemporaryDirectory() as importtempdir:
                zipobject.extractall(importtempdir.name)

                classifier = estimator.Classifier(uniquemodelid, modeldir)
                classifier.import_classifier(importtempdir.name)

    return 'Ok', 200


@app.route('/deletemodel', methods=['POST'])
@check_access
@setup_base_dir(storage, False, False)
def deletemodel():
    # All processing is delegated to delete_dir as it is file system dependant.
    storage.delete_dir()
    return 'Ok', 200


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-visible', action='store_true',
                        help='Allow connections from beyond localhost')
    parser.add_argument('-p', '--port', default=5000, type=int,
                        help='listen on this port')
    parser.add_argument('--debug-mode', action='store_true',
                        help='enable debug features (unsafe in production)')
    args = parser.parse_args()

    if not args.world_visible:
        app.run(debug=True, port=args.port)
    else:
        app.run(host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()
