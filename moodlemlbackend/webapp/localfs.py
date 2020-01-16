import shutil
import os

from functools import wraps, update_wrapper

from moodlemlbackend.webapp.util import get_request_value


# We can not set LocalFS_setup_base_dir as a nested class because they
# have problems to access the outer class.'''


class LocalFS(object):

    def get_localbasedir(self):
        if self.localbasedir is None:
            raise Exception('localbasedir is not set')

        return self.localbasedir

    def set_localbasedir(self, basedir):
        self.localbasedir = basedir

    def get_model_dir(self, hashkey, fetch_model=False):
        '''Returns the model dir in the local fs for the provided key.

        fetch_model param is ignored here.'''

        uniquemodelid = get_request_value('uniqueid')
        dirhash = get_request_value(hashkey)

        # The dir in the local filesystem is namespaced by uniquemodelid and
        # the dirhash which determines where the results should be stored.
        modeldir = os.path.join(self.get_localbasedir(),
                                uniquemodelid, dirhash)

        return modeldir

    def delete_dir(self):

        uniquemodelid = get_request_value('uniqueid')

        # All files related to this version of the model in moodle are in
        # /uniquemodelid.
        modeldir = os.path.join(self.get_localbasedir(), uniquemodelid)

        if os.path.exists(modeldir):
            # The directory may not exist.
            shutil.rmtree(modeldir, True)


class LocalFS_setup_base_dir(object):

    def __init__(self, storage, fetch_model, push_model):
        '''Checks that the local directory is set in ENV.

        fetch_model and push_model are ignored in local_fs.'''

        self.storage = storage

        if "MOODLE_MLBACKEND_PYTHON_DIR" not in os.environ:
            raise IOError(
                'Set env MOODLE_MLBACKEND_PYTHON_DIR to an existing dir')

        localbasedir = os.environ["MOODLE_MLBACKEND_PYTHON_DIR"]

        if not os.path.exists(localbasedir):
            raise IOError(
                'The base dir does not exist. ' +
                'Set env MOODLE_MLBACKEND_PYTHON_DIR to an existing dir')

        os.access(localbasedir, os.W_OK)

        storage.set_localbasedir(localbasedir)

    def __call__(self, f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            '''Execute the decorated function passing the call args.'''

            update_wrapper(self, f)
            return f(*args, **kwargs)
        return wrapper
