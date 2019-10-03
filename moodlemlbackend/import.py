"""Import module"""

import sys
import json
import time

from moodlemlbackend.processor import estimator


def import_classifier():
    """Imports a trained classifier."""

    modelid = sys.argv[1]
    directory = sys.argv[2]

    classifier = estimator.Classifier(modelid, directory)
    classifier.import_classifier(sys.argv[3])

    print('Ok')
    # An exception will be thrown before if it can be imported.
    print('Ok')
    sys.exit(0)

import_classifier()
