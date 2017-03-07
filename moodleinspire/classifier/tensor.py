import math

import numpy as np
from sklearn import preprocessing
import tensorflow as tf

class TF(object):

    def __init__(self, n_features, n_classes, n_epoch, batch_size, starter_learning_rate, tensor_logdir):

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.starter_learning_rate = starter_learning_rate
        self.n_features = n_features
        self.n_classes = n_classes
        self.tensor_logdir = tensor_logdir

        self.x = None
        self.y_ = None
        self.y = None
        self.z = None

        self.build_graph()

        self.start_session()

    def  __getstate__(self):
        state = self.__dict__.copy()
        del state['x']
        del state['y_']
        del state['y']
        del state['z']
        del state['train_step']
        del state['sess']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.build_graph()
        self.start_session()

    def build_graph(self):

        # Placeholders for input values.
        self.x = tf.placeholder(tf.float64, [None, self.n_features], name='x')
        self.y_ = tf.placeholder(tf.float64, [None, self.n_classes], name='dataset-y')

        # Variables for computed stuff, we need to initialise them now.
        W = tf.Variable(tf.zeros([self.n_features, self.n_classes], dtype=tf.float64), name='weights')
        b = tf.Variable(tf.zeros([self.n_classes], dtype=tf.float64), name='bias')

        # Predicted y.
        self.z = tf.matmul(self.x, W) + b
        self.y = tf.nn.softmax(self.z)

        cross_entropy = - tf.reduce_sum(self.y_ * tf.log(tf.clip_by_value(self.y, -1.0, 1.0)))
        loss = tf.reduce_mean(cross_entropy)

        # Calculate decay_rate.
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step,
            100, 0.96, staircase=True)

        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    def start_session(self):

        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_session(self):
        return self.sess

    def fit(self, X, y):

        n_examples, unused = X.shape

        # 1 column per value so will be easier later to make this work with multiple classes.
        #y = y.astype(float)
        y = preprocessing.MultiLabelBinarizer().fit_transform(y.reshape(len(y), 1))

        # floats division otherwise we get 0 if n_examples is lower than the
        # batch size and minimum 1 iteration.
        iterations = int(math.ceil(float(n_examples) / float(self.batch_size)))

        for e in range(self.n_epoch):
            for i in range(iterations):

                offset = i * self.batch_size
                it_end = offset + self.batch_size
                if it_end > n_examples:
                    it_end = n_examples - 1

                batch_xs = X[offset:it_end]
                batch_ys = y[offset:it_end]
                self.sess.run(self.train_step, {self.x: batch_xs, self.y_: batch_ys})

    def predict(self, x):
        return self.sess.run(tf.argmax(self.y, 1), {self.x: x})

    def predict_proba(self, x):
        return self.sess.run(self.z, {self.x: x})

