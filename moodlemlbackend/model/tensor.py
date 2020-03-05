"""Tensorflow classifier"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn import preprocessing
import tensorflow as tf
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

MODEL_DTYPE = 'float32'

class TF(object):
    """Tensorflow classifier"""

    def __init__(self, n_features, n_classes, n_epoch, batch_size,
                 tensor_logdir, initial_weights=False):

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.n_features = n_features

        # Based on the number of features although we need a reasonable
        # minimum.
        self.n_hidden = max(4, int(n_features / 3))
        self.n_hidden_layers = 2
        self.n_classes = n_classes
        self.tensor_logdir = tensor_logdir

        self.build_graph(initial_weights)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        # We also remove this as it depends on the run.
        del state['tensor_logdir']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.build_graph()

    def set_tensor_logdir(self, tensor_logdir):
        """Sets tensorflow logs directory

        Needs to be set separately as it depends on the
        run, it can not be restored."""

        self.tensor_logdir = tensor_logdir

    def build_graph(self, initial_weights=False):
        """Builds the computational graph without feeding any data in"""
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        inputs = tf.keras.Input(shape=(self.n_features,), dtype=MODEL_DTYPE)
        prev = inputs
        for i in range(self.n_hidden_layers):
            h = tf.keras.layers.Dense(self.n_hidden,
                                      name=f'hidden_{i+1}',
                                      activation=tf.nn.relu,
                                      dtype=MODEL_DTYPE)(prev)
            prev = h
        outputs = tf.keras.layers.Dense(self.n_classes,
                                        activation=tf.nn.softmax,
                                        dtype=MODEL_DTYPE)(prev)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.compile()

    def compile(self):
        self.model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['CategoricalAccuracy',
                     tf.keras.metrics.AUC()],
        )

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.compile()

    def fit(self, X, y, log_run=True):
        """Fit the model to the provided data"""
        y = preprocessing.MultiLabelBinarizer().fit_transform(
            y.reshape(len(y), 1))
        y = y.astype(np.float32)

        kwargs = {}
        if log_run:
            cb = tf.compat.v1.keras.callbacks.TensorBoard(
                log_dir=self.tensor_logdir,
                #histogram_freq=1,
                #write_graph=True,
                #write_grads=True,
                write_images=True,
            )
            kwargs['callbacks'] = [cb]

        history = self.model.fit(X, y,
                                 self.batch_size,
                                 self.n_epoch,
                                 verbose=2,
                                 validation_split=0.1,  # XXX
                                 **kwargs
        )

        # The history.history dict contains lists of numpy.float64
        # values which don't work well with json. We need to turn them
        # into floats.
        ret = {}
        for k, v in history.history.items():
            ret[k] = [float(x) for x in v]

        return ret

    def predict(self, x):
        """Find the index of the most probable class."""
        y = self.model.predict(x)
        return tf.keras.backend.eval(tf.argmax(y, 1))

    def predict_proba(self, x):
        """Find the probability distribution over all classes."""
        return self.model.predict(x)
