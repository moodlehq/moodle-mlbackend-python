"""Export module"""

from __future__ import print_function
import sys

from moodlemlbackend.processor import estimator


def export_classifier():
    """Exports the classifier."""

    modelid = sys.argv[1]
    directory = sys.argv[2]

    binary_classifier = estimator.Binary(modelid, directory)
    exportdir = binary_classifier.export_classifier(sys.argv[3])
    if exportdir:
        print(exportdir)
        sys.exit(0)

    sys.exit(1)

export_classifier()
