import os
import logging
import warnings
import time
import math

import numpy as np
from sklearn.utils import shuffle
from sklearn.externals import joblib

class Classifier(object):

    PERSIST_FILENAME = 'classifier.pkl'

    OK = 0
    GENERAL_ERROR = 1
    NO_DATASET = 2
    EVALUATE_LOW_SCORE = 4
    EVALUATE_NOT_ENOUGH_DATA = 8

    def __init__(self, modelid, directory, log_into_file=True):

        self.classes = None

        self.modelid = modelid

        self.runid = str(int(time.time()))

        self.persistencedir = os.path.join(directory, 'classifier')
        if os.path.isdir(self.persistencedir) == False:
            if os.makedirs(self.persistencedir) == False:
                raise OSError('Directory ' + self.persistencedir + ' can not be created.')

        # We define logsdir even though we may not use it.
        self.logsdir = os.path.join(directory, 'logs', self.get_runid())
        if os.path.isdir(self.logsdir):
            raise OSError('Directory ' + self.logsdir + ' already exists.')
        if os.makedirs(self.logsdir) == False:
            raise OSError('Directory ' + self.logsdir + ' can not be created.')

        # Logging.
        self.log_into_file = log_into_file
        logfile = self.get_log_filename()
        logging.basicConfig(filename=logfile,level=logging.DEBUG)
        warnings.showwarning = self.warnings_to_log

        self.X = None
        self.y = None

        self.reset_rates()

        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=5)
        np.set_printoptions(threshold=np.inf)
        np.seterr(all='raise')

    @staticmethod
    def warnings_to_log(message, category, filename, lineno, file=None):
       logging.warning('%s:%s: %s:%s' % (filename, lineno, category.__name__, message))

    def get_runid(self):
        return self.runid


    def get_log_filename(self):
        if self.log_into_file == False:
            return False
        return os.path.join(self.logsdir, 'info.log')

    def load_classifier(self):
        classifier_filepath = os.path.join(self.persistencedir, Classifier.PERSIST_FILENAME)
        return joblib.load(classifier_filepath)

    def store_classifier(self, trained_classifier):
        classifier_filepath = os.path.join(self.persistencedir, Classifier.PERSIST_FILENAME)
        joblib.dump(trained_classifier, classifier_filepath)

    def get_labelled_samples(self, filepath):

        # We skip 3 rows of metadata.
        samples = np.genfromtxt(filepath, delimiter=',', dtype='float', skip_header=3, missing_values='', filling_values=False)
        samples = shuffle(samples)

        # All columns but the last one.
        X = np.array(samples[:,0:-1])

        # Only the last one and as integer.
        y = np.array(samples[:,-1:]).astype(int)

        return [X, y]


    def get_unlabelled_samples(self, filepath):

        # The first column is the sample id with its time range index as a string. The file contains 3 rows of metadata.
        sampleids = np.genfromtxt(filepath, delimiter=',', dtype=np.str, skip_header=3, missing_values='', filling_values=False, usecols=0)

        # We don't know the number of columns, we can only get them all and discard the first one.
        samples = np.genfromtxt(filepath, delimiter=',', dtype=float, skip_header=3, missing_values='', filling_values=False)
        x = samples[:, 1:]

        return [sampleids, x]

    def check_classes_balance(self, counts):
        for item1 in counts:
            for item2 in counts:
                if item1 > (item2 * 3):
                    return 'Provided classes are very unbalanced, predictions may not be accurate.'
        return False

    def limit_value(self, value, lower_bounds, upper_bounds):
        # Limits the value by lower and upper boundaries.
        if value < (lower_bounds - 1):
            return lower_bounds
        elif value > (upper_bounds + 1):
            return upper_bounds
        else:
            return value

    def scale_x(self):
	"""Deprecated, input data should already come scaled."""

        # Limit values to 2 standard deviations from the mean in order
        # to avoid extreme values.
        devs = np.std(self.X, axis=0) * 2
        means = np.mean(self.X, axis=0)
        lower_bounds = means - devs
        upper_bounds = means + devs

        # Switch to an array by features to loop through bounds.
        Xf = np.rollaxis(self.X, axis=1)
        for i, values in enumerate(Xf):
            Xf[i] = [self.limit_value(x, lower_bounds[i], upper_bounds[i]) for x in Xf[i]]

        # Return to an array by samples.
        self.X = np.rollaxis(Xf, axis=1)

        return self.X

    def reset_rates(self):
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.phis = []
