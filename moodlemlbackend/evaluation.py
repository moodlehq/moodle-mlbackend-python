"""Models' evaluation module"""

import sys
import json
import time

from moodlemlbackend.processor import estimator


def evaluation():
    """Evaluates the provided dataset."""

    # Missing arguments.
    if len(sys.argv) < 7:
        result = dict()
        result['runid'] = str(int(time.time()))
        result['status'] = estimator.GENERAL_ERROR
        result['info'] = ['Missing arguments, you should set:\
    - The model unique identifier\
    - The directory to store all generated outputs\
    - The training file\
    - The minimum score (from 0 to 1) to consider the model as valid (defaults to 0.6)\
    - The minimum deviation to accept the model as valid (defaults to 0.02)\
    - The number of times the evaluation will run (defaults to 100)\
    Received: ' + ' '.join(sys.argv)]

        print(json.dumps(result))
        sys.exit(result['status'])

    modelid = sys.argv[1]
    directory = sys.argv[2]
    dataset = sys.argv[3]

    classifier = estimator.Classifier(modelid, directory, dataset)

    if len(sys.argv) > 7:
        trained_model_dir = sys.argv[7]
    else:
        trained_model_dir = False

    result = classifier.evaluate_dataset(dataset,
                                                float(sys.argv[4]),
                                                float(sys.argv[5]),
                                                int(sys.argv[6]),
                                                trained_model_dir)

    print(json.dumps(result))
    sys.exit(result['status'])

evaluation()
