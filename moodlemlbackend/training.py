"""Training module"""

from __future__ import print_function
import sys
import json
import time

from moodlemlbackend.processor import estimator
from moodlemlbackend.processor import binary

def training():
    """Delegates training to train_dataset."""

    # Missing arguments.
    if len(sys.argv) < 4:
        result = dict()
        result['runid'] = str(int(time.time()))
        result['status'] = estimator.Classifier.GENERAL_ERROR
        result['info'] = ['Missing arguments, you should set:\
    - The model unique identifier\
    - The directory to store all generated outputs\
    - The training file\
    Received: ' + ' '.join(sys.argv)]

        print(json.dumps(result))
        sys.exit(result['status'])

    modelid = sys.argv[1]
    directory = sys.argv[2]

    # Sklearn binary classifier - logistic regression.
    #binary_classifier = binary.Sklearn(modelid, directory)
    # TensorFlow binary classifier - NN.
    binary_classifier = binary.TensorFlow(modelid, directory)

    result = binary_classifier.train_dataset(sys.argv[3])

    print(json.dumps(result))
    sys.exit(result['status'])

training()
