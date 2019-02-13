"""Abstract estimator module, will contain just 1 class."""

from __future__ import division

import math
import logging
import time
import warnings
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

from ..model import tensor
from .. import chart

from sklearn.utils import shuffle
from sklearn.externals import joblib


OK = 0
GENERAL_ERROR = 1
NO_DATASET = 2
LOW_SCORE = 4
NOT_ENOUGH_DATA = 8

PERSIST_FILENAME = 'classifier.pkl'
EXPORT_MODEL_FILENAME = 'model.json'


class Estimator(object):
    """Abstract estimator class"""

    def __init__(self, modelid, directory):

        self.X = None
        self.y = None

        self.modelid = modelid

        # Using milliseconds to avoid collisions.
        self.runid = str(int(time.time() * 1000))

        self.persistencedir = os.path.join(directory, 'classifier')
        if os.path.isdir(self.persistencedir) is False:
            if os.makedirs(self.persistencedir) is False:
                raise OSError('Directory ' + self.persistencedir +
                              ' can not be created.')

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

        self.reset_metrics()

        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=5)
        np.set_printoptions(threshold=np.inf)
        np.seterr(all='raise')

    @staticmethod
    def warnings_to_log(message, category, filename, lineno):
        """showwarnings overwritten"""
        logging.warning('%s:%s: %s:%s', filename, lineno,
                        category.__name__, message)

    def get_runid(self):
        """Returns the run id"""
        return self.runid

    def load_classifier(self, model_dir=False):
        """Loads a previously stored classifier"""

        if model_dir is False:
            model_dir = self.persistencedir

        classifier_filepath = os.path.join(
            model_dir, PERSIST_FILENAME)
        return joblib.load(classifier_filepath)

    def store_classifier(self, trained_classifier):
        """Stores the provided classifier"""
        classifier_filepath = os.path.join(
            self.persistencedir, PERSIST_FILENAME)
        joblib.dump(trained_classifier, classifier_filepath)

    @staticmethod
    def get_labelled_samples(filepath):
        """Extracts labelled samples from the provided data file"""

        # We skip 3 rows of metadata.
        samples = np.genfromtxt(filepath, delimiter=',', dtype='float',
                                skip_header=3, missing_values='',
                                filling_values=False)
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

        # The first column is the sample id with its time range index
        # as a string. The file contains 3 rows of metadata.
        sampleids = np.genfromtxt(filepath, delimiter=',', dtype=np.str,
                                  skip_header=3, missing_values='',
                                  filling_values=False,
                                  usecols=0)

        # We don't know the number of columns, we can only get them all and
        # discard the first one.
        samples = np.genfromtxt(filepath, delimiter=',',
                                dtype=float, skip_header=3,
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
                    return 'Provided classes are very unbalanced, ' + \
                        'predictions may not be accurate.'
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


class Binary(Estimator):
    """Binary classifier"""

    def __init__(self, modelid, directory):

        super(Binary, self).__init__(modelid, directory)

        self.aucs = []
        self.classes = [1, 0]

        self.roc_curve_plot = None

        self.tensor_logdir = self.get_tensor_logdir()
        if os.path.isdir(self.tensor_logdir) is False:
            if os.makedirs(self.tensor_logdir) is False:
                raise OSError('Directory ' + self.tensor_logdir +
                              ' can not be created.')

    def get_classifier(self, X, y, initial_weights=False,
                       force_n_features=False):
        """Gets the classifier"""

        n_epoch = 50
        batch_size = 1000
        starter_learning_rate = 0.5

        if force_n_features is not False:
            n_features = force_n_features
        else:
            _, n_features = X.shape

        n_classes = 2

        return tensor.TF(n_features, n_classes, n_epoch, batch_size,
                         starter_learning_rate, self.get_tensor_logdir(),
                         initial_weights=initial_weights)

    def get_tensor_logdir(self):
        """Returns the directory to store tensorflow framework logs"""
        return os.path.join(self.logsdir, 'tensor')

    def store_classifier(self, trained_classifier):
        """Stores the classifier and saves a checkpoint of the tensors state"""

        # Store the graph state.
        saver = tf.train.Saver()
        sess = trained_classifier.get_session()

        path = os.path.join(self.persistencedir, 'model.ckpt')
        saver.save(sess, path)

        # Also save it to the logs dir to see the embeddings.
        path = os.path.join(self.get_tensor_logdir(), 'model.ckpt')
        saver.save(sess, path)

        # Save the class data.
        super(Binary, self).store_classifier(trained_classifier)

    def export_classifier(self, exporttmpdir):
        if self.classifier_exists():
            classifier = self.load_classifier()
        else:
            return False

        export_vars = {}

        # Get all the variables in in initialise-vars scope.
        sess = classifier.get_session()
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='initialise-vars'):
            # Converting to list as numpy arrays can't be serialised.
            export_vars[var.op.name] = var.eval(sess).tolist()

        # Append the number of features.
        export_vars['n_features'] = classifier.get_n_features()

        vars_file_path = os.path.join(exporttmpdir, EXPORT_MODEL_FILENAME)
        with open(vars_file_path, 'w') as vars_file:
            json.dump(export_vars, vars_file)

        return exporttmpdir

    def import_classifier(self, importdir):

        model_vars_filepath = os.path.join(importdir,
                                           EXPORT_MODEL_FILENAME)

        with open(model_vars_filepath) as vars_file:
            import_vars = json.load(vars_file)

        n_features = import_vars['n_features']

        classifier = self.get_classifier(False, False,
                                         initial_weights=import_vars,
                                         force_n_features=n_features)

        self.store_classifier(classifier)

    def load_classifier(self, model_dir=False):
        """Loads a previously trained classifier and restores its state"""

        if model_dir is False:
            model_dir = self.persistencedir

        classifier = super(Binary, self).load_classifier(model_dir)

        classifier.set_tensor_logdir(self.get_tensor_logdir())

        # Now restore the graph state.
        saver = tf.train.Saver()
        path = os.path.join(model_dir, 'model.ckpt')
        saver.restore(classifier.get_session(), path)
        return classifier

    def train(self, X_train, y_train, classifier=False):
        """Train the classifier with the provided training data"""

        if classifier is False:
            # Init the classifier.
            classifier = self.get_classifier(X_train, y_train)

        # Fit the training set. y should be an array-like.
        classifier.fit(X_train, y_train[:, 0])

        # Returns the trained classifier.
        return classifier

    def classifier_exists(self):
        """Checks if there is a previously stored classifier"""

        classifier_dir = os.path.join(self.persistencedir,
                                      PERSIST_FILENAME)
        return os.path.isfile(classifier_dir)

    def train_dataset(self, filepath):
        """Train the model with the provided dataset"""

        [self.X, self.y] = self.get_labelled_samples(filepath)

        if len(np.unique(self.y)) < 2:
            # We need samples belonging to all different classes.
            result = dict()
            result['status'] = NOT_ENOUGH_DATA
            result['info'] = []
            result['errors'] = 'Training data needs to include ' + \
                'samples belonging to all classes'
            return result

        # Load the loaded model if it exists.
        if self.classifier_exists():
            classifier = self.load_classifier()
        else:
            # Not previously trained.
            classifier = False

        trained_classifier = self.train(self.X, self.y, classifier)

        self.store_classifier(trained_classifier)

        result = dict()
        result['status'] = OK
        result['info'] = []
        return result

    def predict_dataset(self, filepath):
        """Predict labels for the provided dataset"""

        [sampleids, x] = self.get_unlabelled_samples(filepath)

        if self.classifier_exists() is False:
            result = dict()
            result['status'] = NO_DATASET
            result['info'] = ['Provided model have not been trained yet']
            return result

        classifier = self.load_classifier()

        # Prediction and calculated probability of each of the labels.
        y_proba = classifier.predict_proba(x)
        y_pred = classifier.predict(x)
        # Probabilities of the predicted response being correct.
        probabilities = y_proba[range(len(y_proba)), y_pred]

        result = dict()
        result['status'] = OK
        result['info'] = []
        # First column sampleids, second the prediction and third how
        # reliable is the prediction (from 0 to 1).
        result['predictions'] = np.vstack((sampleids,
                                           y_pred,
                                           probabilities)).T.tolist()

        return result

    def evaluate_dataset(self, filepath, min_score=0.6,
                         accepted_deviation=0.02, n_test_runs=100,
                         trained_model_dir=False):
        """Evaluate the model using the provided dataset"""

        [self.X, self.y] = self.get_labelled_samples(filepath)

        # Classes balance check.
        counts = []
        y_array = np.array(self.y.T[0])
        counts.append(np.count_nonzero(y_array))
        counts.append(len(y_array) - np.count_nonzero(y_array))
        logging.info('Number of samples by y value: %s', str(counts))
        balanced_classes = self.check_classes_balance(counts)
        if balanced_classes is not False:
            logging.warning(balanced_classes)

        # Check that we have samples belonging to all classes.
        if counts[0] == 0 or counts[1] == 0:
            result = dict()
            result['runid'] = int(self.get_runid())
            result['status'] = GENERAL_ERROR
            result['info'] = ['The provided dataset does not contain ' +
                              'samples for each class']
            return result

        # ROC curve.
        self.roc_curve_plot = chart.RocCurve(self.logsdir, 2)

        if trained_model_dir is not False:
            # Load the trained model in the provided path and evaluate it.
            trained_model_dir = os.path.join(trained_model_dir, 'classifier')
            classifier = self.load_classifier(trained_model_dir)
            self.rate_prediction(classifier, self.X, self.y)

        else:
            # Evaluate the model by training the ML algorithm multiple times.

            for _ in range(0, n_test_runs):

                # Split samples into training set and test set (80% - 20%)
                X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                                    self.y,
                                                                    test_size=0.2)

                classifier = self.train(X_train, y_train)

                self.rate_prediction(classifier, X_test, y_test)

        # Store the roc curve.
        logging.info("Figure stored in " + self.roc_curve_plot.store())

        # Return results.
        result = self.get_evaluation_results(min_score, accepted_deviation)

        # Add the run id to identify it in the caller.
        result['runid'] = int(self.get_runid())

        logging.info("Accuracy: %.2f%%", result['accuracy'] * 100)
        logging.info("AUC: %.2f%%", result['auc'])
        logging.info("Precision (predicted elements that are real): %.2f%%",
                     result['precision'] * 100)
        logging.info("Recall (real elements that are predicted): %.2f%%",
                     result['recall'] * 100)
        logging.info("Score: %.2f%%", result['score'] * 100)
        logging.info("AUC standard desviation: %.4f", result['auc_deviation'])

        return result

    def rate_prediction(self, classifier, X_test, y_test):
        """Rate a trained classifier with test data"""

        # Calculate scores.
        y_score = self.get_score(classifier, X_test, y_test[:, 0])
        y_pred = classifier.predict(X_test)

        # Transform it to an array.
        y_test = y_test.T[0]

        # Calculate accuracy, sensitivity and specificity.
        [acc, prec, rec, phi] = self.calculate_metrics(y_test == 1,
                                                       y_pred == 1)
        self.accuracies.append(acc)
        self.precisions.append(prec)
        self.recalls.append(rec)
        self.phis.append(phi)

        # ROC curve calculations.
        fpr, tpr, _ = roc_curve(y_test, y_score)

        # When the amount of samples is small we can randomly end up
        # having just one class instead of examples of each, which
        # triggers a "UndefinedMetricWarning: No negative samples in
        # y_true, false positive value should be meaningless"
        # and returning NaN.
        if math.isnan(fpr[0]) or math.isnan(tpr[0]):
            return

        self.aucs.append(auc(fpr, tpr))

        # Draw it.
        self.roc_curve_plot.add(fpr, tpr, 'Positives')

    @staticmethod
    def get_score(classifier, X_test, y_test):
        """Returns the trained classifier score"""

        probs = classifier.predict_proba(X_test)

        n_samples = len(y_test)

        # Calculated probabilities of the correct response.
        return probs[range(n_samples), y_test]

    @staticmethod
    def calculate_metrics(y_test_true, y_pred_true):
        """Calculates confusion matrix metrics"""

        test_p = y_test_true
        test_n = np.invert(test_p)

        pred_p = y_pred_true
        pred_n = np.invert(pred_p)

        pp = np.count_nonzero(test_p)
        nn = np.count_nonzero(test_n)
        tp = np.count_nonzero(test_p * pred_p)
        tn = np.count_nonzero(test_n * pred_n)
        fn = np.count_nonzero(test_p * pred_n)
        fp = np.count_nonzero(test_n * pred_p)

        accuracy = (tp + tn) / (pp + nn)
        if tp != 0 or fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        if tp != 0 or fn != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if denominator != 0:
            phi = ((tp * tn) - (fp * fn)) / math.sqrt(denominator)
        else:
            phi = 0

        return [accuracy, precision, recall, phi]

    def reset_metrics(self):

        super(Binary, self).reset_metrics()

        self.aucs = []

        # ROC curve.
        self.roc_curve_plot = chart.RocCurve(self.logsdir, 2)

    def get_evaluation_results(self, min_score, accepted_deviation):
        """Returns the evaluation results after all iterations"""

        avg_accuracy = np.mean(self.accuracies)
        avg_precision = np.mean(self.precisions)
        avg_recall = np.mean(self.recalls)
        avg_aucs = np.mean(self.aucs)
        avg_phi = np.mean(self.phis)

        # Phi goes from -1 to 1 we need to transform it to a value between
        # 0 and 1 to compare it with the minimum score provided.
        score = (avg_phi + 1) / 2

        result = dict()
        result['auc'] = avg_aucs
        result['accuracy'] = avg_accuracy
        result['precision'] = avg_precision
        result['recall'] = avg_recall
        result['auc_deviation'] = np.std(self.aucs)
        result['score'] = score
        result['min_score'] = min_score
        result['accepted_deviation'] = accepted_deviation

        result['dir'] = self.logsdir
        result['status'] = OK
        result['info'] = []

        # If deviation is too high we may need more records to report if
        # this model is reliable or not.
        auc_deviation = np.std(self.aucs)
        if auc_deviation > accepted_deviation:
            result['info'].append('The evaluation results varied too much, ' +
                                  'we need more samples to check if this ' +
                                  'model is valid. Model deviation = ' +
                                  str(auc_deviation) +
                                  ', accepted deviation = ' +
                                  str(accepted_deviation))
            result['status'] = NOT_ENOUGH_DATA

        if score < min_score:
            result['info'].append('The evaluated model prediction accuracy ' +
                                  'is not very good. Model score = ' +
                                  str(score) + ', minimum score = ' +
                                  str(min_score))
            result['status'] = LOW_SCORE

        if auc_deviation > accepted_deviation and score < min_score:
            result['status'] = LOW_SCORE + NOT_ENOUGH_DATA

        result['info'].append('Launch TensorBoard from command line by ' +
                              'typing: tensorboard --logdir=\'' +
                              self.get_tensor_logdir() + '\'')

        return result
