"""Prediction module"""

import sys
import json
import time

from moodlemlbackend.processor import estimator


def prediction():
    """Returns predictions for a given dataset."""

    # Missing arguments.
    if len(sys.argv) < 4:
        result = dict()
        result['runid'] = str(int(time.time()))
        result['status'] = estimator.GENERAL_ERROR
        result['info'] = ['Missing arguments, you should set:\
    - The model unique identifier\
    - The directory to store all generated outputs\
    - The file with samples to predict\
    Received: ' + ' '.join(sys.argv)]

        print(json.dumps(result))
        sys.exit(result['status'])

    modelid = sys.argv[1]
    directory = sys.argv[2]
    dataset = sys.argv[3]

    # TensorFlow binary classifier - NN.
    classifier = estimator.Classifier(modelid, directory, dataset)

    result = classifier.predict_dataset(dataset)

    print(json.dumps(result))
    sys.exit(result['status'])

prediction()
