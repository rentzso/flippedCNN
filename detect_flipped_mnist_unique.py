from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np
import tensorflow as tf

class FlipDetector(object):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    all_images = np.concatenate((mnist.train.images, mnist.test.images, mnist.validation.images))
    all_labels = np.concatenate((mnist.train.labels, mnist.test.labels, mnist.validation.labels))
    
    def __init__(self, digits):
        self.digits = digits
        self.datasets = self._compute_datasets(digits)
    
    def _compute_datasets(self, digits):
        # select only the images for the input digits
        filter_ = np.in1d(self.all_labels, digits)
        filtered_images = self.all_images[filter_]
        # flip all the images selected
        flipped = np.flip(filtered_images.reshape([-1, 28, 28]), axis=2)
        flipped = flipped.reshape([-1, 784])
        # flip some of the images
        true_ar = np.ones(len(filtered_images), dtype=bool)
        false_ar = np.zeros(len(filtered_images), dtype=bool)
        is_flipped = np.where( np.random.rand(len(filtered_images)) > 0.5, true_ar, false_ar)
        flipped_chosen = flipped[is_flipped]
        filtered_images_chosen = filtered_images[np.logical_not(is_flipped)]
        images = np.concatenate((filtered_images_chosen, flipped_chosen))
        # create labels for images: 
        # - originals have one in the first column and zero in the second, 
        # - flipped zero in the first, one in the second
        labels_col_1 = np.concatenate(( np.ones(len(filtered_images_chosen)), np.zeros(len(flipped_chosen)))).reshape([-1, 1])
        labels_col_2 = np.concatenate(( np.zeros(len(filtered_images_chosen)), np.ones(len(flipped_chosen)))).reshape([-1, 1])
        labels = np.concatenate((labels_col_1, labels_col_2), axis=1)
        assert len(labels) == len(images), 'labels and images have different length'
        # shuffle inputs and labels randomly but in unison
        p = np.random.permutation(len(images))
        images = images[p]
        labels = labels[p]
        # roughly using the same mnist proportions for train, test and validation size
        train_size = 11*labels.shape[0]/14
        test_size = 2*labels.shape[0]/14
        validation_size = labels.shape[0]/14
        # build the datasets
        train = DataSet(images[:train_size], labels[:train_size], reshape=False)
        test = DataSet(images[train_size:train_size + test_size], labels[train_size:train_size + test_size], reshape=False)
        validation = DataSet(images[train_size + test_size:], labels[train_size + test_size:], reshape=False)
        # the returned datasets has the same class as the mnist datasets
        return base.Datasets(train=train, test=test, validation=validation)
    
    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
        
    def prepare_cnn(self):
        # adapted from https://www.tensorflow.org/get_started/mnist/pros
        
        #inputs
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2])
        x_image = tf.reshape(self.x, [-1,28,28,1])
        # first layer
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        # second layer        
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        # densely connected layer
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        # readout layer
        W_fc2 = self.weight_variable([1024, 2])
        b_fc2 = self.bias_variable([2])

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def run_cnn(self, tf_sess, iterations=1000):
        tf_sess.run(tf.initialize_all_variables())
        for i in range(iterations):
            batch = self.datasets.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = tf_sess.run(
                    self.accuracy,
                    feed_dict={self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            tf_sess.run(self.train_step, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
    
    def results(self, tf_sess, dataset):
        print("accuracy on the dataset %g"%tf_sess.run(
            self.accuracy,
            feed_dict={
                self.x: dataset.images, self.y_: dataset.labels, self.keep_prob: 1.0
            }))
        y_est = tf_sess.run(self.y_conv, feed_dict={self.x:dataset.images, self.keep_prob: 1.0})
        y_est = np.argmax(y_est, 1)
        y = np.argmax(dataset.labels, 1)
        errors = dataset.images[y!=y_est]
        error_labels = y[y!=y_est]
        return DataSet(errors, error_labels, reshape=False)
        