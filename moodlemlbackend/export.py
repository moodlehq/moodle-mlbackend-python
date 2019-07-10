"""Export module"""

import sys

from moodlemlbackend.processor import estimator


def export_classifier():
    """Exports the classifier."""

    modelid = sys.argv[1]
    directory = sys.argv[2]

    classifier = estimator.Classifier(modelid, directory)
    exportdir = classifier.export_classifier(sys.argv[3])
    if exportdir:
        print(exportdir)
        sys.exit(0)

    sys.exit(1)

export_classifier()
