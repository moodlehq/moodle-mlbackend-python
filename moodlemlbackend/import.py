"""Import module"""

from __future__ import print_function
import sys
import json
import time

from moodlemlbackend.processor import estimator


def import_classifier():
    """Imports a trained classifier."""

    modelid = sys.argv[1]
    directory = sys.argv[2]

    binary_classifier = estimator.Binary(modelid, directory)
    binary_classifier.import_classifier(sys.argv[3])

    # An exception will be thrown before if it can be imported.
    sys.exit(0)

import_classifier()
