"""Tensorflow classifier"""

import math
import os

from sklearn import preprocessing
import tensorflow as tf

class TF(object):
    """Tensorflow classifier"""

    def __init__(self, n_features, n_classes, n_epoch, batch_size,
                 starter_learning_rate, tensor_logdir):

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
        self.loss = None

        self.build_graph()

        self.start_session()

        # During evaluation we process the same dataset multiple times, could could store
        # each run result to the user but results would be very similar we only do it once
        # making it simplier to understand and saving disk space.
        if os.listdir(self.tensor_logdir) == []:
            self.log_run = True
            self.init_logging()
        else:
            self.log_run = False



    def  __getstate__(self):
        state = self.__dict__.copy()
        del state['x']
        del state['y_']
        del state['y']
        del state['z']
        del state['train_step']
        del state['sess']

        del state['file_writer']
        del state['merged']

        # We also remove this as it depends on the run.
        del state['tensor_logdir']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.build_graph()
        self.start_session()

    def set_tensor_logdir(self, tensor_logdir):
        '''Needs to be set separately as it depends on the run, it can not be restored.'''
        self.tensor_logdir = tensor_logdir
        try:
            self.file_writer
            self.merged
        except AttributeError:
            # Init logging if logging vars are not defined.
            self.init_logging()

    def build_graph(self):
        """Builds the computational graph without feeding any data in"""

        # Placeholders for input values.
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float64, [None, self.n_features], name='x')
            self.y_ = tf.placeholder(tf.float64, [None, self.n_classes], name='dataset-y')

        # Variables for computed stuff, we need to initialise them now.
        with tf.name_scope('weights'):
            W = tf.Variable(tf.zeros([self.n_features, self.n_classes], dtype=tf.float64),
                            name='weights')
            b = tf.Variable(tf.zeros([self.n_classes], dtype=tf.float64), name='bias')

        # Predicted y.
        with tf.name_scope('activation'):
            self.z = tf.matmul(self.x, W) + b
            tf.summary.histogram('predicted_values', self.z)
            self.y = tf.nn.softmax(self.z)
            tf.summary.histogram('activations', self.y)

        with tf.name_scope('loss'):
            cross_entropy = - tf.reduce_sum(self.y_ * tf.log(tf.clip_by_value(self.y, -1.0, 1.0)))
            loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar("loss", loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # Calculate decay_rate.
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step,
                                                   100, 0.96, staircase=False)
        tf.summary.scalar("learning_rate", learning_rate)

        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    def start_session(self):
        """Starts the session"""

        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def init_logging(self):
        """Starts logging the tensors state"""
        self.file_writer = tf.summary.FileWriter(self.tensor_logdir, self.sess.graph)
        self.merged = tf.summary.merge_all()

    def get_session(self):
        """Return the session"""
        return self.sess

    def fit(self, X, y):
        """Fits provided data into the session"""

        n_examples, _ = X.shape

        # 1 column per value so will be easier later to make this work with multiple classes.
        y = preprocessing.MultiLabelBinarizer().fit_transform(y.reshape(len(y), 1))

        # floats division otherwise we get 0 if n_examples is lower than the
        # batch size and minimum 1 iteration.
        iterations = int(math.ceil(float(n_examples) / float(self.batch_size)))

        index = 0
        for _ in range(self.n_epoch):
            for j in range(iterations):

                offset = j * self.batch_size
                it_end = offset + self.batch_size
                if it_end > n_examples:
                    it_end = n_examples - 1

                batch_xs = X[offset:it_end]
                batch_ys = y[offset:it_end]

                if self.log_run:
                    _, summary = self.sess.run([self.train_step, self.merged],
                                               {self.x: batch_xs, self.y_: batch_ys})
                    # Add the summary data to the file writer.
                    self.file_writer.add_summary(summary, index)
                else:
                    self.sess.run(self.train_step, {self.x: batch_xs, self.y_: batch_ys})

                index = index + 1

    def predict(self, x):
        """Returns predictions"""
        return self.sess.run(tf.argmax(self.y, 1), {self.x: x})

    def predict_proba(self, x):
        """Returns predicted probabilities"""
        return self.sess.run(self.z, {self.x: x})
