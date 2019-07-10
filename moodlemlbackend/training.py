"""Training module"""

import sys
import json
import time

from moodlemlbackend.processor import estimator


def training():
    """Trains a ML classifier."""

    # Missing arguments.
    if len(sys.argv) < 4:
        result = dict()
        result['runid'] = str(int(time.time()))
        result['status'] = estimator.GENERAL_ERROR
        result['info'] = ['Missing arguments, you should set:\
    - The model unique identifier\
    - The directory to store all generated outputs\
    - The training file\
    Received: ' + ' '.join(sys.argv)]

        print(json.dumps(result))
        sys.exit(result['status'])

    modelid = sys.argv[1]
    directory = sys.argv[2]
    dataset = sys.argv[3]

    classifier = estimator.Classifier(modelid, directory, dataset)

    result = classifier.train_dataset(dataset)

    print(json.dumps(result))
    sys.exit(result['status'])

training()
