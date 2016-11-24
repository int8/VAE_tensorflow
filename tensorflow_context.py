import tensorflow as tf

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
