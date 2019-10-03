import logging
import shutil
import tempfile
import os
import zipfile

from functools import wraps, update_wrapper

import boto3
from botocore.exceptions import ClientError

from moodlemlbackend.webapp.util import get_request_value, zipdir


# This will be set overwritten below.
localbasedir = None


class S3(object):

    def get_localbasedir(self):
        if self.localbasedir is None:
            raise Exception('localbasedir is not set')

        return self.localbasedir

    def set_localbasedir(self, basedir):
        self.localbasedir = basedir

    def get_model_dir(self, hashkey, fetch_model=False):
        '''Returns the model dir in the local fs for the provided key.

        Syncs the model with S3 if required.'''

        # No need to include uniquemodelid here, because this is all lives in
        # a temp dir in the local file system.
        modeldir = os.path.join(self.get_localbasedir(), hashkey)

        if fetch_model:

            s3 = boto3.client('s3')

            # Download the files for the provided uniquemodelid + modelhash
            bucketname = os.environ["MOODLE_MLBACKEND_PYTHON_S3_BUCKET_NAME"]
            objectkey = self.object_key(hashkey)

            # TODO Check if we should be using TemporaryFile instead.
            classifierzip = tempfile.NamedTemporaryFile()
            classifierdir = os.path.join(modeldir, 'classifier')
            try:
                s3.download_fileobj(bucketname, objectkey, classifierzip)

                if os.path.getsize(classifierzip.name) > 0:
                    with zipfile.ZipFile(classifierzip, 'r') as zipobject:

                        # The classifier directory is automatically created in
                        # moodlemlbackend.estimator but we need to create it
                        # before that point as we want to copy the classifier
                        # from S3.
                        try:
                            os.makedirs(classifierdir)
                        except FileExistsError:
                            # It can exist in some cases.
                            pass
                        zipobject.extractall(classifierdir)
            except ClientError:
                # No worries, it may perfectly not exist.
                pass

        return modeldir

    def delete_dir(self):

        s3 = boto3.resource('s3')

        bucketname = os.environ["MOODLE_MLBACKEND_PYTHON_S3_BUCKET_NAME"]
        bucket = s3.Bucket(bucketname)

        # Objectkey will equal uniquemodelid so we delete all files matching
        # uniquemodelid/ namespace.
        objectkey = self.object_key(False)

        bucket.objects.filter(Prefix=objectkey + '/').delete()

    def object_key(self, hashkey=False):

        uniquemodelid = get_request_value('uniqueid')

        if hashkey is False:
            return uniquemodelid

        dirhash = get_request_value(hashkey)
        return os.path.join(uniquemodelid, dirhash)


class S3_setup_base_dir(object):
    '''Sets the localbasedir to /tmp'''

    def __init__(self, storage, fetch_model, push_model):
        '''Sets the base dir to a temp directory.

        It fetches the requested model from s3 if required.'''

        self.storage = storage
        self.fetch_model = fetch_model
        self.push_model = push_model

        # It is our responsibility to delete this directory. However, we are
        # relying on the OS to delete it if there is any exception during the
        # course of the request.
        self.storage.set_localbasedir(tempfile.mkdtemp())

    def __call__(self, f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            '''Execute the decorated function.

            Upload the model to s3 if required.'''

            update_wrapper(self, f)

            self.modeldir = self.storage.get_model_dir(
                'dirhash', fetch_model=self.fetch_model)

            # Execute the requested action.
            funcreturn = f(*args, **kwargs)

            if self.push_model is True:
                # Push the model to s3.

                s3 = boto3.client('s3')

                classifierdir = os.path.join(self.modeldir, 'classifier')

                # Copy the classifier in the model dir to S3.
                updatedclassifierzip = tempfile.NamedTemporaryFile()
                zipdir(classifierdir, updatedclassifierzip)

                # We are only interested in the model we just trained.
                bucketname = os.environ[
                    "MOODLE_MLBACKEND_PYTHON_S3_BUCKET_NAME"]
                objectkey = self.storage.object_key('dirhash')

                # Upload to S3.
                try:
                    s3.upload_file(
                        updatedclassifierzip.name, bucketname, objectkey)
                except ClientError as e:
                    # We don't want the error details in moodle as they could
                    # contain sensitive information.
                    logging.error('Error uploading the model to S3: ' + str(e))
                    return 'Can\'t upload classifier to S3.', 500

                # TODO Think about copying the new logs to S3.

            # It is our responsibility to delete tmp directories created with
            # mkdtemp
            shutil.rmtree(self.storage.get_localbasedir(), True)

            # Now that the files are copied back to S3 we can return f's
            # Response.
            return funcreturn

        return wrapper
