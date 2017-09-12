"""Abstract estimator module, will contain just 1 class."""

import os
import logging
import warnings
import time

import numpy as np
from sklearn.utils import shuffle
from sklearn.externals import joblib

class Classifier(object):
    """Abstract estimator class"""

    PERSIST_FILENAME = 'classifier.pkl'

    OK = 0
    GENERAL_ERROR = 1
    NO_DATASET = 2
    LOW_SCORE = 4
    NOT_ENOUGH_DATA = 8

    def __init__(self, modelid, directory):

        self.classes = None

        self.modelid = modelid

        # Using milliseconds to avoid collisions.
        self.runid = str(int(time.time() * 1000))

        self.persistencedir = os.path.join(directory, 'classifier')
        if os.path.isdir(self.persistencedir) is False:
            if os.makedirs(self.persistencedir) is False:
                raise OSError('Directory ' + self.persistencedir + ' can not be created.')

        # We define logsdir even though we may not use it.
        self.logsdir = os.path.join(directory, 'logs', self.get_runid())
        if os.path.isdir(self.logsdir):
            raise OSError('Directory ' + self.logsdir + ' already exists.')
        if os.makedirs(self.logsdir) is False:
            raise OSError('Directory ' + self.logsdir + ' can not be created.')

        # Logging.
        logfile = os.path.join(self.logsdir, 'info.log')
        logging.basicConfig(filename=logfile, level=logging.DEBUG)
        warnings.showwarning = self.warnings_to_log

        self.X = None
        self.y = None

        self.reset_metrics()

        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=5)
        np.set_printoptions(threshold=np.inf)
        np.seterr(all='raise')


    @staticmethod
    def warnings_to_log(message, category, filename, lineno):
        """showwarnings overwritten"""
        logging.warning('%s:%s: %s:%s', filename, lineno, category.__name__, message)


    def get_runid(self):
        """Returns the run id"""
        return self.runid


    def load_classifier(self):
        """Loads a previously stored classifier"""
        classifier_filepath = os.path.join(self.persistencedir, Classifier.PERSIST_FILENAME)
        return joblib.load(classifier_filepath)

    def store_classifier(self, trained_classifier):
        """Stores the provided classifier"""
        classifier_filepath = os.path.join(self.persistencedir, Classifier.PERSIST_FILENAME)
        joblib.dump(trained_classifier, classifier_filepath)

    @staticmethod
    def get_labelled_samples(filepath):
        """Extracts labelled samples from the provided data file"""

        # We skip 3 rows of metadata.
        samples = np.genfromtxt(filepath, delimiter=',', dtype='float', skip_header=3,
                                missing_values='', filling_values=False)
        samples = shuffle(samples)

        # This is a single sample dataset, genfromtxt returns the samples
        # as a one dimension array, we don't want that.
        if samples.ndim == 1:
            samples = np.array([samples])

        # All columns but the last one.
        X = np.array(samples[:, 0:-1])

        # Only the last one and as integer.
        y = np.array(samples[:, -1:]).astype(int)

        return [X, y]

    @staticmethod
    def get_unlabelled_samples(filepath):
        """Extracts unlabelled samples from the provided data file"""

        # The first column is the sample id with its time range index as a string.
        # The file contains 3 rows of metadata.
        sampleids = np.genfromtxt(filepath, delimiter=',', dtype=np.str,
                                  skip_header=3, missing_values='', filling_values=False,
                                  usecols=0)

        # We don't know the number of columns, we can only get them all and discard the first one.
        samples = np.genfromtxt(filepath, delimiter=',', dtype=float, skip_header=3,
                                missing_values='', filling_values=False)

        # This is a single sample dataset, genfromtxt returns the samples
        # as a one dimension array, we don't want that.
        if samples.ndim == 1:
            samples = np.array([samples])

        x = samples[:, 1:]

        return [sampleids, x]

    @staticmethod
    def check_classes_balance(counts):
        """Checks that the dataset contains enough samples of each class"""
        for item1 in counts:
            for item2 in counts:
                if item1 > (item2 * 3):
                    return 'Provided classes are very unbalanced, predictions may not be accurate.'
        return False

    @staticmethod
    def limit_value(value, lower_bounds, upper_bounds):
        """Limits the value by lower and upper boundaries"""
        if value < (lower_bounds - 1):
            return lower_bounds
        elif value > (upper_bounds + 1):
            return upper_bounds
        else:
            return value

    def reset_metrics(self):
        """Resets the class metrics"""
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.phis = []
