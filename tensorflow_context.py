import tensorflow as tf
import prettytensor as pt

class TensorflowContext:

    def __init__(self):
        self.sess = tf.Session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()

    def save(self, filename):
        saver = tf.train.Saver()
        saver.save(self.sess, filename)

    def load(self, filename):
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)

    def prettytensor_scope(self):
        return pt.defaults_scope(activation_fn=tf.nn.relu, batch_normalize=True, learned_moments_update_rate=0.0003, variance_epsilon=0.001, scale_after_normalization=True)

    def train_scope(self):
        return tf.variable_scope("vaemodel")

    def sampling_scope(self):
        # its important to reuse existing variables when sampling
        return tf.variable_scope("vaemodel", reuse=True)
