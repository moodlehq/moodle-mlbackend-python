import math

import numpy as np
from sklearn import preprocessing
import tensorflow as tf

class TF(object):

    def __init__(self, n_epoch, batch_size, starter_learning_rate, final_learning_rate):

        self.sess = tf.Session()

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.starter_learning_rate = starter_learning_rate
        self.final_learning_rate = final_learning_rate

        self.x = None
        self.y_ = None
        self.z = None

    def fit(self, X, y):

        # TODO Using 2 classes. Extra argument required if we want
        # this to work with more than 2 classes
        n_classes = 2

        n_examples, n_features = X.shape

        # floats division otherwise we get 0 if n_examples is lower than the
        # batch size and minimum 1 iteration.
        iterations = int(math.ceil(float(n_examples) / float(self.batch_size)))

        total_iterations = self.n_epoch * iterations

        # 1 column per value so will be easier later to make this work with multiple classes.
        #y = y.astype(float)
        y = preprocessing.MultiLabelBinarizer().fit_transform(y.reshape(len(y), 1))

        # Placeholders for input values.
        self.x = tf.placeholder(tf.float64, [None, n_features], name='x')
        self.y_ = tf.placeholder(tf.float64, [None, n_classes], name='dataset-y')

        # Variables for computed stuff, we need to initialise them now.
        W = tf.Variable(tf.zeros([n_features, n_classes], dtype=tf.float64), name='weights')
        b = tf.Variable(tf.zeros([n_classes], dtype=tf.float64), name='bias')

        # Predicted y.
        self.z = tf.matmul(self.x, W) + b
        self.y = tf.nn.softmax(self.z)

        cross_entropy = - tf.reduce_sum(self.y_ * tf.log(tf.clip_by_value(self.y, -1.0, 1.0)))
        loss = tf.reduce_mean(cross_entropy)

        # Calculate decay_rate.
        learning_rate = self.calculate_decay_rate(total_iterations)

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        for e in range(self.n_epoch):
            for i in range(iterations):

                offset = i * self.batch_size
                it_end = offset + self.batch_size
                if it_end > n_examples:
                    it_end = n_examples - 1

                batch_xs = X[offset:it_end]
                batch_ys = y[offset:it_end]
                self.sess.run(train_step, {self.x: batch_xs, self.y_: batch_ys})

                #correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
                #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                #print(self.sess.run(accuracy, feed_dict={self.x: X, self.y_: y}))

    def calculate_decay_rate(self, total_iterations):

        decay_rate = math.pow(self.final_learning_rate / self.starter_learning_rate,
            (1. / float(total_iterations)))

        # Learning rate decreasing over time.
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step,
            total_iterations, decay_rate, staircase=True)

        return learning_rate

    def predict(self, x):
        return self.sess.run(tf.argmax(self.y, 1), {self.x: x})

    def predict_proba(self, x):
        return self.sess.run(self.z, {self.x: x})

