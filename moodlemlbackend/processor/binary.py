"""Binary classification module"""

from __future__ import division

import math
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegressionCV
import tensorflow as tf

from . import estimator
from ..classifier import tensor
from .. import chart

class Sklearn(estimator.Classifier):
    """sciki-learn based binary classification manager"""

    def __init__(self, modelid, directory):

        super(Sklearn, self).__init__(modelid, directory)

        self.aucs = []
        self.classes = [1, 0]

        self.roc_curve_plot = None

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

        classifier_dir = os.path.join(self.persistencedir, estimator.Classifier.PERSIST_FILENAME)
        return os.path.isfile(classifier_dir)

    def train_dataset(self, filepath):
        """Train the model with the provided dataset
        TODO Move this to Classifier and make it multiple classes compatible."""

        [self.X, self.y] = self.get_labelled_samples(filepath)

        if len(np.unique(self.y)) < 2:
            # We need samples belonging to all different classes.
            result = dict()
            result['status'] = estimator.Classifier.NOT_ENOUGH_DATA
            result['info'] = []
            result['errors'] = 'Training data needs to include samples belonging to all classes'
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
        result['status'] = estimator.Classifier.OK
        result['info'] = []
        return result


    def predict_dataset(self, filepath):
        """Predict labels for the provided dataset
        TODO Move this to Classifier and make it multiple classes compatible."""

        [sampleids, x] = self.get_unlabelled_samples(filepath)

        if self.classifier_exists() is False:
            result = dict()
            result['status'] = estimator.Classifier.NO_DATASET
            result['info'] = ['Provided model have not been trained yet']
            return result

        classifier = self.load_classifier()

        # Prediction and calculated probability of each of the labels.
        y_proba = classifier.predict_proba(x)
        y_pred = classifier.predict(x)
        # Probabilities of the predicted response being correct.
        probabilities = y_proba[range(len(y_proba)), y_pred]

        result = dict()
        result['status'] = estimator.Classifier.OK
        result['info'] = []
        # First column sampleids, second the prediction and third how
        # reliable is the prediction (from 0 to 1).
        result['predictions'] = np.vstack((sampleids, y_pred, probabilities)).T.tolist()

        return result


    def evaluate_dataset(self, filepath, min_score=0.6, accepted_deviation=0.02, n_test_runs=100):
        """Evaluate the model using the provided dataset
        TODO Move this to Classifier and make it multiple classes compatible."""

        [self.X, self.y] = self.get_labelled_samples(filepath)

        # Classes balance check.
        counts = []
        y_array = np.array(self.y.T[0])
        counts.append(np.count_nonzero(y_array))
        counts.append(len(y_array) - np.count_nonzero(y_array))
        logging.info('Number of samples by y value: %s', str(counts))
        balanced_classes = self.check_classes_balance(counts)
        if balanced_classes != False:
            logging.warning(balanced_classes)

        # Check that we have samples belonging to all classes.
        if counts[0] == 0 or counts[1] == 0:
            result = dict()
            result['runid'] = int(self.get_runid())
            result['status'] = estimator.Classifier.GENERAL_ERROR
            result['info'] = ['The provided dataset does not contain samples for each class']
            return result

        # ROC curve.
        self.roc_curve_plot = chart.RocCurve(self.logsdir, 2)

        # Learning curve.
        try:
            self.store_learning_curve()
        except ValueError:
            # The learning curve cross-validation process may trigger a
            # ValueError if all examples in a data chunk belong to the same
            # class, which is likely to happen when the number of samples
            # is small.
            logging.info('Learning curve generation skipped, not enough samples')

        for _ in range(0, n_test_runs):

            # Split samples into training set and test set (80% - 20%)
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)

            classifier = self.train(X_train, y_train)

            self.rate_prediction(classifier, X_test, y_test)

        # Store the roc curve.
        logging.info("Figure stored in " + self.roc_curve_plot.store())

        # Return results.
        result = self.get_evaluation_results(min_score, accepted_deviation)

        # Add the run id to identify it in the caller.
        result['runid'] = int(self.get_runid())

        logging.info("Accuracy: %.2f%%", result['accuracy'] * 100)
        logging.info("Precision (predicted elements that are real): %.2f%%",
                     result['precision'] * 100)
        logging.info("Recall (real elements that are predicted): %.2f%%", result['recall'] * 100)
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
        [acc, prec, rec, phi] = self.calculate_metrics(y_test == 1, y_pred == 1)
        self.accuracies.append(acc)
        self.precisions.append(prec)
        self.recalls.append(rec)
        self.phis.append(phi)

        # ROC curve calculations.
        fpr, tpr, _ = roc_curve(y_test, y_score)

        # When the amount of samples is small we can randomly end up having just
        # one class instead of examples of each, which triggers a "UndefinedMetricWarning:
        # No negative samples in y_true, false positive value should be meaningless"
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
        result['status'] = estimator.Classifier.OK
        result['info'] = []

        # If deviation is too high we may need more records to report if
        # this model is reliable or not.
        auc_deviation = np.std(self.aucs)
        if auc_deviation > accepted_deviation:
            result['info'].append('The evaluation results varied too much, we need more samples '
                                  + 'to check if this model is valid. Model deviation = ' +
                                  str(auc_deviation) + ', accepted deviation = ' +
                                  str(accepted_deviation))
            result['status'] = estimator.Classifier.NOT_ENOUGH_DATA

        if score < min_score:
            result['info'].append('The evaluated model prediction accuracy is not very good.'
                                  + ' Model score = ' + str(score) + ', minimum score = ' +
                                  str(min_score))
            result['status'] = estimator.Classifier.LOW_SCORE

        if auc_deviation > accepted_deviation and score < min_score:
            result['status'] = estimator.Classifier.LOW_SCORE + \
                estimator.Classifier.NOT_ENOUGH_DATA

        return result


    def store_learning_curve(self):
        """Stores the learning curve of this evaluation"""

        learning_curve = chart.LearningCurve(self.logsdir)
        learning_curve.set_classifier(self.get_classifier(self.X, self.y))
        lc_filepath = learning_curve.store(self.X, self.y)
        logging.info("Figure stored in " + lc_filepath)


    def get_classifier(self, X, y):
        """Returns a logistic regression classifier after cross-validation"""

        solver = 'liblinear'
        multi_class = 'ovr'

        if hasattr(self, 'C') is False:

            # Cross validation - to select the best constants.
            lgcv = LogisticRegressionCV(solver=solver, multi_class=multi_class)
            lgcv.fit(X, y[:, 0])

            if len(lgcv.C_) == 1:
                C = lgcv.C_[0]
            else:
                # Chose the best C = the class with more samples.
                # Ideally multiclass problems will be multinomial.
                [_, counts] = np.unique(y[:, 0], return_counts=True)
                C = lgcv.C_[np.argmax(counts)]
                logging.info('From all classes best C values (%s), %f has been selected',
                             str(lgcv.C_), C)
            logging.info("Best C: %f", C)

        return LogisticRegression(solver=solver, tol=1e-1, C=C)


    def reset_metrics(self):
        super(Sklearn, self).reset_metrics()
        self.aucs = []

        # ROC curve.
        self.roc_curve_plot = chart.RocCurve(self.logsdir, 2)


class TensorFlow(Sklearn):
    """Binary classifier using tensorflow"""

    def __init__(self, modelid, directory):

        super(TensorFlow, self).__init__(modelid, directory)

        self.tensor_logdir = self.get_tensor_logdir()
        if os.path.isdir(self.tensor_logdir) is False:
            if os.makedirs(self.tensor_logdir) is False:
                raise OSError('Directory ' + self.tensor_logdir + ' can not be created.')


    def get_classifier(self, X, y):
        """Gets the classifier
        TODO In future we don't want to hardcode 2 classes"""

        n_epoch = 50
        batch_size = 1000
        starter_learning_rate = 0.5

        _, n_features = X.shape

        n_classes = 2

        return tensor.TF(n_features, n_classes, n_epoch, batch_size, starter_learning_rate,
                         self.get_tensor_logdir())

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
        super(TensorFlow, self).store_classifier(trained_classifier)

    def load_classifier(self):
        """Loads a previously trained classifier and restores its tensors state"""

        classifier = super(TensorFlow, self).load_classifier()
        classifier.set_tensor_logdir(self.get_tensor_logdir())

        # Now restore the graph state.
        saver = tf.train.Saver()
        path = os.path.join(self.persistencedir, 'model.ckpt')
        saver.restore(classifier.get_session(), path)
        return classifier

    def store_learning_curve(self):
        pass

    def get_evaluation_results(self, min_score, accepted_deviation):

        results = super(TensorFlow, self).get_evaluation_results(min_score, accepted_deviation)
        results['info'].append('Launch TensorBoard from command line by typing: '
                               + 'tensorboard --logdir=\'' + self.get_tensor_logdir() + '\'')

        return results
