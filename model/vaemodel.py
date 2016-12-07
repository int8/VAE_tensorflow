import tensorflow as tf
import prettytensor as pt
from common.tensorflow_context import TensorflowContext
import numpy as np
from progressbar import ProgressBar, Percentage, Bar

class VaeAutoencoderSampler(TensorflowContext):

    def __init__(self, decoder, model_path):
        TensorflowContext.__init__(self)
        self.decoder = decoder

        with self.prettytensor_scope as scope:
            with tf.variable_scope("vae"):
                self.sample = self.decoder.generate_network_random_sample()
            self.load(model_path)

    def generate_random_sample(self, epsilon):
        sample = self.sess.run(self.sample, {self.decoder.epsilon: epsilon})
        return sample

    def walk_between_points(self, a, b, number_of_steps):
        points = [a] + [np.add(a * (1 - 1. * i/number_of_steps) , b * (1. * i/number_of_steps))  for i in range(1, number_of_steps)] + [b]
        samples = []
        for point in points:
            sample = self.sess.run(self.sample, {self.decoder.epsilon: point})
            samples.append(sample)
        return samples

class VaeAutoencoderTrainer(TensorflowContext):

    def __init__(self, encoder, decoder, hdf5reader):
        TensorflowContext.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.hdf5reader = hdf5reader

        # compose autoencoder + error function + sampling routine
        with self.prettytensor_scope:
            with tf.variable_scope("vae"):
                self.reconstruction, self.mu, self.stddev = self.decoder.genereate_network_output(self.encoder.get_network_output())
                self.build_error_function()

    def build_error_function(self, eps = 1e-9):

        self.prob_reconstruction_error = -tf.reduce_sum(
            self.encoder.input_data * tf.log(self.reconstruction + eps) + (1.0 - self.encoder.input_data) * tf.log(1.0 - self.reconstruction + eps),
            reduction_indices = [1,2,3]
        )

        # alternatively one could use the following reconstruction error component
        # self.euc_reconstruction_error = tf.reduce_sum(tf.pow(self.encoder.input_data - self.reconstruction + eps,2))

        self.vae_error = - 0.5 * tf.reduce_sum(
            1 + tf.log(tf.square(self.stddev + eps)) -  tf.square(self.mu) - tf.square(self.stddev),
            reduction_indices = 1
        )

        self.error_function = tf.reduce_mean(self.vae_error + self.prob_reconstruction_error)

    def train(self, epochs, batch_size, learning_rate, save_to=None):

        self.train_step = pt.apply_optimizer(tf.train.AdamOptimizer(learning_rate, epsilon=1), losses = [self.error_function])
        init = tf.initialize_all_variables()
        self.sess.run(init)
        pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=epochs).start()
        while self.get_epoch() < epochs:
            input_data = self.hdf5reader.next()
            _, loss_value = self.sess.run(
                [self.train_step, self.error_function],
                {
                    self.encoder.input_data: input_data
                }
            )
            pbar.update(self.get_epoch())
        pbar.finish()

    def get_epoch(self):
        return self.hdf5reader.get_cycles()
