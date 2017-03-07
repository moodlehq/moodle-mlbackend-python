import sys
import json

from ..processor import estimator
from ..processor import binary

def main():

    # Missing arguments.
    if len(sys.argv) < 4:
        result = dict()
        result['runid'] = str(int(time.time()))
        result['status'] = estimator.Classifier.GENERAL_ERROR
        result['errors'] = ['Missing arguments, you should set:\
    - The model unique identifier\
    - The directory to store all generated outputs\
    - The file with samples to predict\
    Received: ' + ' '.join(sys.argv)]

        # Add the provided unique id.
        if len(sys.argv) > 1:
            result['modelid'] = sys.argv[1]

        print(json.dumps(result))
        sys.exit(result['status'])

    modelid = sys.argv[1]
    directory = sys.argv[2]

    # Sklearn binary classifier - logistic regression.
    #binary_classifier = binary.Sklearn(modelid, directory)
    # TensorFlow binary classifier - NN.
    binary_classifier = binary.TensorFlow(modelid, directory)
    # TensorFlow binary classifier - logistic regression.
    #binary_classifier = binary.Skflow(modelid, directory)
    # TensorFlow binary classifier - deep neural network.
    #binary_classifier = binary.DNN(modelid, directory)

    result = binary_classifier.predict_dataset(sys.argv[3])

    print(json.dumps(result))
    sys.exit(result['status'])

main()
