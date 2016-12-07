import tensorflow as tf
import prettytensor as pt

class TensorflowContext:

    def __init__(self, model_path = None):

        self.sess = tf.Session()
        self.prettytensor_scope = pt.defaults_scope(activation_fn=tf.nn.relu, batch_normalize=True, learned_moments_update_rate=0.0003, variance_epsilon=0.001, scale_after_normalization=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()

    def load(self, model_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(model_path))

    def save(self, model_path):
        saver = tf.train.Saver()
        saver.save(self.sess, model_path + '/output')
