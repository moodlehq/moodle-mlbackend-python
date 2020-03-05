"""Abstract estimator module, will contain just 1 class."""
import csv

import math
import logging
import time
import warnings
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Flask deals with logging and this conflicts with tensorflow initialising
# absl.logging
# https://github.com/abseil/abseil-py/issues/102
# https://github.com/abseil/abseil-py/issues/99
import absl.logging
absl.logging._warn_preinit_stderr = 0

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils import shuffle
import tensorflow as tf

from ..model import tensor
from .. import chart


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

OK = 0
GENERAL_ERROR = 1
NO_DATASET = 2
LOW_SCORE = 4
NOT_ENOUGH_DATA = 8

PERSIST_FILENAME = 'classifier.pkl'
EXPORT_MODEL_FILENAME = 'model.json'

TARGET_BATCH_SIZE = 1000


class Estimator(object):
    """Abstract estimator class"""

    def __init__(self, modelid, directory):

        self.X = None
        self.y = None
        self.variable_columns = None

        self.modelid = modelid

        # Using milliseconds to avoid collisions.
        self.runid = str(int(time.time() * 1000))

        self.persistencedir = os.path.join(directory, 'classifier')

        os.makedirs(self.persistencedir, exist_ok=True)

        # We define logsdir even though we may not use it.
        self.logsdir = os.path.join(directory, 'logs', self.get_runid())
        os.makedirs(self.logsdir)

        # Logging.
        logfile = os.path.join(self.logsdir, 'info.log')
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=logfile, level=logging.DEBUG)
        warnings.showwarning = self.warnings_to_log

        self.reset_metrics()

        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=5)
        np.set_printoptions(threshold=np.inf)
        np.seterr(all='raise')

    @staticmethod
    def warnings_to_log(message, category, filename, lineno, file=None,
                        line=None):
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

        classifier = joblib.load(classifier_filepath)
        self.variable_columns = getattr(classifier, 'variable_columns', None)
        path = os.path.join(model_dir, 'model.ckpt')
        classifier.load(path)

        return classifier

    def store_classifier(self, trained_classifier):
        """Stores the provided classifier"""
        trained_classifier.variable_columns = self.variable_columns
        classifier_filepath = os.path.join(
            self.persistencedir, PERSIST_FILENAME)
        joblib.dump(trained_classifier, classifier_filepath)

    @staticmethod
    def get_labelled_samples(filepath):
        """Extracts labelled samples from the provided data file"""

        # We skip 3 rows of metadata.
        samples = np.genfromtxt(filepath, delimiter=',', dtype=np.float32,
                                skip_header=3, missing_values='',
                                filling_values=0.0)
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
                                dtype=np.float32, skip_header=3,
                                missing_values='', filling_values=False)

        # This is a single sample dataset, genfromtxt returns the samples
        # as a one dimension array, we don't want that.
        if samples.ndim == 1:
            samples = np.array([samples])

        x = samples[:, 1:]

        return [sampleids, x]

    @staticmethod
    def get_metadata(filepath):
        with open(filepath) as datafile:
            file_iterator = csv.reader(datafile, delimiter='\n', quotechar='"')
            row_count = 0
            for row in file_iterator:
                row_count += 1
                if row_count == 1:
                    data_header = [x for x in csv.reader(
                        row, delimiter=',', quotechar='"')][0]
                    classes_index = data_header.index("targetclasses")
                    features_index = data_header.index("nfeatures")
                if row_count == 2:
                    info_row = [x for x in csv.reader(
                        row, delimiter=',', quotechar='"')][0]
                    target_classes = json.loads(info_row[classes_index])
                    return {
                        "n_classes": len(target_classes),
                        "classes": target_classes,
                        "n_features": int(info_row[features_index])
                    }

    @staticmethod
    def check_classes_balance(counts):
        """Checks that the dataset contains enough samples of each class"""
        if max(counts) > min(counts) * 3:
            return ('Provided classes are very unbalanced, '
                    'predictions may not be accurate.')

    def reset_metrics(self):
        """Resets the class metrics"""
        self.baccuracies = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []


class Classifier(Estimator):
    """General classifier"""

    def __init__(self, modelid, directory, dataset=None):
        super(Classifier, self).__init__(modelid, directory)

        self.aucs = []
        self.roc_curve_plot = chart.RocCurve(self.logsdir, 2)

        if dataset:
            meta = self.get_metadata(dataset)
            self.n_features = meta['n_features']
            self.classes = meta['classes']
            self.n_classes = meta['n_classes']
            self.is_binary = self.n_classes == 2

        self.tensor_logdir = self.get_tensor_logdir()
        os.makedirs(self.tensor_logdir, exist_ok=True)

    def get_classifier(self, X, y, initial_weights=False):
        """Gets the classifier"""

        try:
            n_rows = X.shape[0]
        except AttributeError:
            # No X during model import.
            # n_rows value does not really matter during import.
            n_rows = 1

        n_batches = (n_rows + TARGET_BATCH_SIZE - 1) // TARGET_BATCH_SIZE
        n_batches = min(n_batches, 10)
        batch_size = (n_rows + n_batches - 1) // n_batches

        # the number of epochs can be smaller if we have a large
        # number of samples. On the other hand it must also be small
        # if we have very few samples, or the model will overfit. What
        # we can say is that with larger batches we need more epochs.
        n_epoch = 40 + batch_size // 20

        n_classes = self.n_classes
        n_features = X.shape[1]

        return tensor.TF(n_features, n_classes, n_epoch, batch_size,
                         self.get_tensor_logdir(),
                         initial_weights=initial_weights)

    def train(self, X_train, y_train, classifier=False, log_run=True):
        """Train the classifier with the provided training data"""

        if classifier is False:
            # Init the classifier.
            classifier = self.get_classifier(X_train, y_train)

        # Fit the training set. y should be an array-like.
        classifier.fit(X_train, y_train[:, 0], log_run=log_run)
        self.store_classifier(classifier)
        # Returns the trained classifier.
        return classifier

    def remove_invariant_columns(self, X):
        if self.variable_columns is None:
            return X
        return X[:,self.variable_columns]

    def find_invariant_columns(self, X):
        if self.variable_columns is not None:
            logging.warning("variable columns have already been found!")
            logging.warning("removing them again would be trouble")
            return
        self.variable_columns = np.nonzero(np.var(X, 0))[0]

    def train_dataset(self, filepath):
        """Train the model with the provided dataset"""
        X, self.y = self.get_labelled_samples(filepath)

        # Load the loaded model if it exists.
        if self.classifier_exists():
            classifier = self.load_classifier()
        else:
            # Not previously trained.
            classifier = False

        self.find_invariant_columns(X)
        self.X = self.remove_invariant_columns(X)

        if len(np.unique(self.y)) < self.n_classes:
            # We need samples belonging to all different classes.
            result = dict()
            result['status'] = NOT_ENOUGH_DATA
            result['info'] = []
            result['errors'] = 'Training data needs to include ' + \
                'samples belonging to all classes'
            return result

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

        x = self.remove_invariant_columns(x)
        # Prediction and calculated probability of each of the labels.
        y_proba = classifier.predict_proba(x)
        y_pred = classifier.predict(x)
        # Probabilities of the predicted response being correct.
        probabilities = y_proba[np.arange(len(y_proba)), y_pred]

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

        X, self.y = self.get_labelled_samples(filepath)
        self.find_invariant_columns(X)
        self.X = self.remove_invariant_columns(X)

        # Classes balance check.
        y_array = np.array(self.y.T[0])
        unique_elements, counts = np.unique(y_array, return_counts=True)

        if not np.array_equal(np.sort(self.classes), np.sort(unique_elements)):
            result = dict()
            result['runid'] = int(self.get_runid())
            result['status'] = GENERAL_ERROR
            result['info'] = ['The labels from the provided dataset do not ' +
                              'match the targetclasses from the header']
            return result

        logging.info('Number of samples by y value: %s', str(counts))
        balanced_classes = self.check_classes_balance(counts)
        if balanced_classes:
            logging.warning(balanced_classes)

        # Check that we have samples belonging to all classes.
        for i in range(len(counts)):
            if counts[i] == 0:
                result = dict()
                result['runid'] = int(self.get_runid())
                result['status'] = GENERAL_ERROR
                result['info'] = ['The provided dataset does not contain ' +
                                  'samples for each class']
                return result

        if trained_model_dir is not False:
            # Load the trained model in the provided path and evaluate it.
            trained_model_dir = os.path.join(trained_model_dir, 'classifier')
            classifier = self.load_classifier(trained_model_dir)
            self.rate_prediction(classifier, self.X, self.y)

        else:
            # Evaluate the model by training the ML algorithm multiple times.
            for i in range(n_test_runs):
                # Split samples into training set and test set (80% - 20%)
                X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                                    self.y,
                                                                    test_size=0.2)
                if len(np.unique(y_train)) < self.n_classes:
                    # We need the input data to match the expected size of the
                    # tensor.
                    continue

                log_run = i == 0
                classifier = self.train(X_train, y_train, log_run=log_run)

                self.rate_prediction(classifier, X_test, y_test)

        # Store the roc curve.
        logging.info("Figure stored in " + self.roc_curve_plot.store())

        # Return results.
        result = self.get_evaluation_results(min_score, accepted_deviation)

        print("score: " + str(result['score']))

        # Add the run id to identify it in the caller.
        result['runid'] = int(self.get_runid())

        if self.is_binary:
            logging.info("AUC: %.2f%%", result['auc'])
            logging.info("AUC standard deviation: %.4f",
                         result['auc_deviation'])

        logging.info("Accuracy: %.2f%%", result['accuracy'] * 100)
        logging.info("Precision (predicted elements that are real): %.2f%%",
                     result['precision'] * 100)
        logging.info("Recall (real elements that are predicted): %.2f%%",
                     result['recall'] * 100)
        logging.info("Score: %.2f%%", result['score'] * 100)
        logging.info("Score standard deviation: %.4f",
                     result['score_deviation'])

        return result

    def rate_prediction(self, classifier, X_test, y_test):
        """Rate a trained classifier with test data"""

        # Calculate scores.
        y_score = self.get_score(classifier, X_test, y_test[:, 0])
        y_pred = classifier.predict(X_test)

        # Transform it to an array.
        y_test = y_test.T[0]

        if self.is_binary:

            try:
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

            except Exception:
                # Nevermind.
                pass

        # Calculate accuracy, sensitivity and specificity.
        [bacc, acc, prec, rec, f1score] = self.calculate_metrics(y_test, y_pred)

        self.baccuracies.append(bacc)
        self.accuracies.append(acc)
        self.precisions.append(prec)
        self.recalls.append(rec)
        self.f1_scores.append(f1score)

    @staticmethod
    def get_score(classifier, X_test, y_test):
        """Returns the trained classifier score"""

        probs = classifier.predict_proba(X_test)
        n_samples = len(y_test)

        # Calculated probabilities of the correct response.
        return probs[range(n_samples), y_test]

    @staticmethod
    def calculate_metrics(y_test, y_pred):
        """Calculates the accuracy metrics"""

        accuracy = accuracy_score(y_test, y_pred)
        baccuracy = balanced_accuracy_score(y_test, y_pred)

        # Wrapping all the scoring function calls in a try & except to prevent
        # the following warning to result in a "TypeError: warnings_to_log()
        # takes 4 positional arguments but 6 were given" when sklearn calls
        # warnings.warn with an "UndefinedMetricWarning:Precision is
        # ill-defined and being set to 0.0 in labels with no predicted
        # samples." message on python 3.7.x
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1score = f1_score(y_test, y_pred, average='weighted')

        return [baccuracy, accuracy, precision, recall, f1score]

    def get_evaluation_results(self, min_score, accepted_deviation):
        """Returns the evaluation results after all iterations"""

        avg_baccuracy = np.mean(self.baccuracies)
        avg_accuracy = np.mean(self.accuracies)
        avg_precision = np.mean(self.precisions)
        avg_recall = np.mean(self.recalls)

        if len(self.f1_scores) > 0:
            score = np.mean(self.f1_scores)
            score_deviation = np.std(self.f1_scores)
        else:
            score = 0
            score_deviation = 1

        result = dict()
        if self.is_binary and len(self.aucs) > 0:
            try:
                result['auc'] = np.mean(self.aucs)
                result['auc_deviation'] = np.std(self.aucs)
            except Exception:
                # No worries.
                result['auc'] = 0.0
                result['auc_deviation'] = 1.0
                pass

        result['balanced accuracy'] = avg_baccuracy
        result['accuracy'] = avg_accuracy
        result['precision'] = avg_precision
        result['recall'] = avg_recall
        result['f1_score'] = score
        result['score_deviation'] = score_deviation
        result['score'] = score
        result['min_score'] = min_score
        result['accepted_deviation'] = accepted_deviation

        result['dir'] = self.logsdir
        result['status'] = OK
        result['info'] = []

        # If deviation is too high we may need more records to report if
        # this model is reliable or not.
        if score_deviation > accepted_deviation:
            result['info'].append('The evaluation results varied too much, ' +
                                  'we might need more samples to check if this ' +
                                  'model is valid. Model deviation = ' +
                                  str(score_deviation) +
                                  ', accepted deviation = ' +
                                  str(accepted_deviation))
            result['status'] = NOT_ENOUGH_DATA

        if score < min_score:
            result['info'].append('The evaluated model prediction accuracy ' +
                                  'is not very good. Model score = ' +
                                  str(score) + ', minimum score = ' +
                                  str(min_score))
            result['status'] = LOW_SCORE

        if score_deviation > accepted_deviation and score < min_score:
            result['status'] = LOW_SCORE + NOT_ENOUGH_DATA

        return result

    def store_classifier(self, trained_classifier):
        """Stores the classifier and saves a checkpoint of the tensors state"""

        # Store the graph state.
        path = os.path.join(self.persistencedir, 'model.ckpt')
        trained_classifier.save(path)
        path = os.path.join(self.get_tensor_logdir(), 'model.ckpt')
        trained_classifier.save(path)
        super().store_classifier(trained_classifier)

    def load_classifier(self, model_dir=False):
        """Loads a previously trained classifier and restores its state"""

        if model_dir is False:
            model_dir = self.persistencedir

        classifier = super(Classifier, self).load_classifier(model_dir)

        classifier.set_tensor_logdir(self.get_tensor_logdir())

        return classifier

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
        export_vars['n_features'] = classifier.n_features
        export_vars['n_classes'] = classifier.n_classes

        vars_file_path = os.path.join(exporttmpdir, EXPORT_MODEL_FILENAME)
        with open(vars_file_path, 'w') as vars_file:
            json.dump(export_vars, vars_file)

        return exporttmpdir

    def import_classifier(self, importdir):

        model_vars_filepath = os.path.join(importdir,
                                           EXPORT_MODEL_FILENAME)

        with open(model_vars_filepath) as vars_file:
            import_vars = json.load(vars_file)

        self.n_features = import_vars['n_features']
        if "n_classes" in import_vars:
            self.n_classes = import_vars['n_classes']
        else:
            self.n_classes = 2

        classifier = self.get_classifier(False, False,
                                         initial_weights=import_vars)

        self.store_classifier(classifier)

    def classifier_exists(self):
        """Checks if there is a previously stored classifier"""
        classifier_file = os.path.join(self.persistencedir,
                                       PERSIST_FILENAME)
        return os.path.isfile(classifier_file)

    def get_tensor_logdir(self):
        """Returns the directory to store tensorflow framework logs"""
        return os.path.join(self.logsdir, 'tensor')
