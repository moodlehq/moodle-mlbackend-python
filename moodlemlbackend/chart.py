"""Charts module"""

import os

import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class LearningCurve(object):
    """scikit-learn Learning curve class"""

    def __init__(self, dirname):
        self.dirname = dirname
        self.classifier = None

    def set_classifier(self, classifier):
        """Set a classifier"""
        self.classifier = classifier

    def store(self, X, y, figure_id=1):
        """Save the learning curve"""

        plt.figure(figure_id)
        plt.xlabel("Training samples")
        plt.ylabel("Error")

        train_sizes, train_scores, test_scores = learning_curve(
            self.classifier, X, y[:, 0])

        train_error_mean = 1 - np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_error_mean = 1 - np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_error_mean + train_scores_std,
                         train_error_mean - train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_error_mean + test_scores_std,
                         test_error_mean - test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_error_mean, 'o-', color="r",
                 label="Training error")
        plt.plot(train_sizes, test_error_mean, 'o-', color="g",
                 label="Cross-validation error")
        plt.legend(loc="best")

        filepath = os.path.join(self.dirname, 'learning-curve.png')
        plt.savefig(filepath, format='png')

        if not os.path.isfile(filepath):
            return False

        return filepath


class RocCurve(object):
    """Class to generate a ROC curve chart through scikit-learn"""

    def __init__(self, dirname, figid):
        self.dirname = dirname
        plt.figure(figid)

    @staticmethod
    def add(fpr, tpr, label):
        """Add data to the chart"""
        plt.plot(fpr, tpr, label=label)

    def store(self):
        """Store the ROC curve"""
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Provided data ROC curve/s')
        plt.legend(loc="lower right")

        filepath = os.path.join(self.dirname, 'roc-curve.png')
        plt.savefig(filepath, format='png')

        return filepath
