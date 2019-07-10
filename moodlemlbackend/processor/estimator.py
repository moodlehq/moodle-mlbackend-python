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
from sklearn.metrics import confusion_matrix
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
    def get_metadata(filepath):
        with open(filepath) as datafile:
            file_iterator = csv.reader(datafile, delimiter='\n', quotechar='"')
            row_count = 0
            for row in file_iterator:
                row_count += 1
                if row_count == 1:
                    data_header = [x for x in csv.reader(row, delimiter=',', quotechar='"')][0]
                    classes_index = data_header.index("targetclasses")
                    features_index = data_header.index("nfeatures")
                if row_count == 2:
                    info_row = [x for x in csv.reader(row, delimiter=',', quotechar='"')][0]
                    target_classes = json.loads(info_row[classes_index])
                    return {
                        "n_classes": len(target_classes),
                        "classes": target_classes,
                        "n_features": int(info_row[features_index])
                    }

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
        self.mccs = []


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
        if os.path.isdir(self.tensor_logdir) is False:
            if os.makedirs(self.tensor_logdir) is False:
                raise OSError('Directory ' + self.tensor_logdir +
                              ' can not be created.')


    def get_classifier(self, X, y, initial_weights=False):
        """Gets the classifier"""

        n_epoch = 50
        batch_size = 1000
        starter_learning_rate = 0.5

        n_classes = self.n_classes
        n_features = self.n_features

        return tensor.TF(n_features, n_classes, n_epoch, batch_size,
                         starter_learning_rate, self.get_tensor_logdir(),
                         initial_weights=initial_weights)

    def train(self, X_train, y_train, classifier=False):
        """Train the classifier with the provided training data"""

        if classifier is False:
            # Init the classifier.
            classifier = self.get_classifier(X_train, y_train)

        # Fit the training set. y should be an array-like.
        classifier.fit(X_train, y_train[:, 0])

        # Returns the trained classifier.
        return classifier

    def train_dataset(self, filepath):
        """Train the model with the provided dataset"""
        [self.X, self.y] = self.get_labelled_samples(filepath)

        if len(np.unique(self.y)) < self.n_classes:
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
        if balanced_classes is not False:
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

        print("score: " + str(result['score']))

        # Add the run id to identify it in the caller.
        result['runid'] = int(self.get_runid())

        if self.is_binary:
            logging.info("AUC: %.2f%%", result['auc'])
            logging.info("AUC standard deviation: %.4f", result['auc_deviation'])
        logging.info("Accuracy: %.2f%%", result['accuracy'] * 100)
        logging.info("Precision (predicted elements that are real): %.2f%%",
                     result['precision'] * 100)
        logging.info("Recall (real elements that are predicted): %.2f%%",
                     result['recall'] * 100)
        logging.info("Score: %.2f%%", result['score'] * 100)
        logging.info("Score standard deviation: %.4f", result['acc_deviation'])

        return result

    def rate_prediction(self, classifier, X_test, y_test):
        """Rate a trained classifier with test data"""

        # Calculate scores.
        y_score = self.get_score(classifier, X_test, y_test[:, 0])
        y_pred = classifier.predict(X_test)

        # Transform it to an array.
        y_test = y_test.T[0]

        if self.is_binary:
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

        # Calculate accuracy, sensitivity and specificity.
        mcc = self.get_mcc(y_test, y_pred)

        [acc, prec, rec] = self.calculate_metrics(y_test == 1, y_pred == 1)

        self.accuracies.append(acc)
        self.precisions.append(prec)
        self.recalls.append(rec)
        self.mccs.append(mcc)

    @staticmethod
    def get_mcc(y_true, y_pred):
        C = confusion_matrix(y_true, y_pred)
        t_sum = C.sum(axis=1, dtype=np.float64)
        p_sum = C.sum(axis=0, dtype=np.float64)
        n_correct = np.trace(C, dtype=np.float64)
        n_samples = p_sum.sum()
        cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
        cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
        cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
        denominator = np.sqrt(cov_ytyt * cov_ypyp)
        if denominator != 0:
            mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
        else:
            return 0.

        if np.isnan(mcc):
            return 0.
        else:
            return mcc

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

        return [accuracy, precision, recall]

    def get_evaluation_results(self, min_score, accepted_deviation):
        """Returns the evaluation results after all iterations"""

        avg_accuracy = np.mean(self.accuracies)
        avg_precision = np.mean(self.precisions)
        avg_recall = np.mean(self.recalls)
        avg_mcc = np.mean(self.mccs)

        # MCC goes from -1 to 1 we need to transform it to a value between
        # 0 and 1 to compare it with the minimum score provided.
        score = (avg_mcc + 1) / 2

        acc_deviation = np.std(self.mccs)
        result = dict()
        if self.is_binary:
            result['auc'] = np.mean(self.aucs)
            result['auc_deviation'] = np.std(self.aucs)
        result['accuracy'] = avg_accuracy
        result['precision'] = avg_precision
        result['recall'] = avg_recall
        result['acc_deviation'] = acc_deviation
        result['score'] = score
        result['min_score'] = min_score
        result['accepted_deviation'] = accepted_deviation

        result['dir'] = self.logsdir
        result['status'] = OK
        result['info'] = []

        # If deviation is too high we may need more records to report if
        # this model is reliable or not.
        if acc_deviation > accepted_deviation:
            result['info'].append('The evaluation results varied too much, ' +
                                  'we might need more samples to check if this ' +
                                  'model is valid. Model deviation = ' +
                                  str(acc_deviation) +
                                  ', accepted deviation = ' +
                                  str(accepted_deviation))
            result['status'] = NOT_ENOUGH_DATA

        if score < min_score:
            result['info'].append('The evaluated model prediction accuracy ' +
                                  'is not very good. Model score = ' +
                                  str(score) + ', minimum score = ' +
                                  str(min_score))
            result['status'] = LOW_SCORE

        if acc_deviation > accepted_deviation and score < min_score:
            result['status'] = LOW_SCORE + NOT_ENOUGH_DATA

        result['info'].append('Launch TensorBoard from command line by ' +
                              'typing: tensorboard --logdir=\'' +
                              self.get_tensor_logdir() + '\'')

        return result

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
        super(Classifier, self).store_classifier(trained_classifier)

    def load_classifier(self, model_dir=False):
        """Loads a previously trained classifier and restores its state"""

        if model_dir is False:
            model_dir = self.persistencedir

        classifier = super(Classifier, self).load_classifier(model_dir)

        classifier.set_tensor_logdir(self.get_tensor_logdir())

        # Now restore the graph state.
        saver = tf.train.Saver()
        path = os.path.join(model_dir, 'model.ckpt')
        saver.restore(classifier.get_session(), path)
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
        export_vars['n_features'] = classifier.get_n_features()
        export_vars['n_classes'] = classifier.get_n_classes()

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

        classifier_dir = os.path.join(self.persistencedir,
                                      PERSIST_FILENAME)
        return os.path.isfile(classifier_dir)

    def get_tensor_logdir(self):
        """Returns the directory to store tensorflow framework logs"""
        return os.path.join(self.logsdir, 'tensor')
