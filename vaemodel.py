import tensorflow as tf
import prettytensor as pt
from tensorflow_context import TensorflowContext
import numpy as np
from scipy.misc import imsave

class VaeAutoencoder(TensorflowContext):

    def __init__(self, encoder, decoder, data_generator, mode = 'train'):
        TensorflowContext.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.data_generator = data_generator

        with self.prettytensor_scope():
            with self.train_scope():
                self.reconstruction, self.mu, self.stddev = self.decoder.genereate_network_output(self.encoder.get_network_output())
                self.build_error_function()
            with self.sampling_scope():
                self.random_sample = self.decoder.generate_network_random_sample(1)

    def build_error_function(self, eps = 1e-9):

        self.prob_reconstruction_error = -tf.reduce_sum(
            self.encoder.input_data * tf.log(self.reconstruction + eps) + (1.0 - self.encoder.input_data) * tf.log(1.0 - self.reconstruction + eps),
            reduction_indices = [1,2,3]
        )

        # alternatively one could use the following reconstruction error component
        # self.euc_reconstruction_error = tf.reduce_sum(tf.pow(self.encoder.input_data - self.reconstruction + epsilon,2))

        self.vae_error = - 0.5 * tf.reduce_sum(
            1 + tf.log(tf.square(self.stddev + eps)) -  tf.square(self.mu) - tf.square(self.stddev),
            reduction_indices = 1
        )

        self.error_function = tf.reduce_mean(self.vae_error + self.prob_reconstruction_error)

    def train(self, epochs, batch_size, learning_rate, save_to=None):

        self.train_step = pt.apply_optimizer(tf.train.AdamOptimizer(learning_rate, epsilon=1), losses = [self.error_function])
        init = tf.initialize_all_variables()
        self.sess.run(init)

        for epoch in range(epochs):
            input_data = next(self.data_generator)
            _, loss_value = self.sess.run(
                [self.train_step, self.error_function],
                {
                    self.encoder.input_data: input_data
                }
            )

            s = self.generate_random_sample()
            imsave('generated/' + str(epoch) + 'randomsample.png', s[0])
        if save_to:
            self.save(save_to)

    def generate_random_sample(self):
        sample = self.sess.run(self.random_sample)
        return sample
