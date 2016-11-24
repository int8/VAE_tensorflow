import tensorflow as tf
import prettytensor as pt
from vae_config import pt_default_scope
from tensorflow_context import TensorflowContext
import numpy as np
from scipy.misc import imsave

class VaeAutoencoder(TensorflowContext):

    def __init__(self, encoder, decoder, data_generator, input_channels, eps, mode = 'train'):
        TensorflowContext.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.data_generator = data_generator
        if mode == 'train':
            with pt_default_scope:
                self.reconstruction, self.mu, self.sig_log_sq = self.decoder.get_network_output(self.encoder.get_network_output())
                self.build_error_function(input_channels, eps)
        elif mode == "sample":
            self.zeros_tensor = tf.zeros([1, self.encoder.representation_size], tf.float32)
            self.random_sample, self.mu, self.sig_log_sq = self.decoder.get_network_output((self.zeros_tensor,self.zeros_tensor) , 1)
        else:
            raise Exception("Unknown mode")

    def build_error_function(self, input_channels, eps):
        self.vae_error = -0.5 * tf.reduce_sum(
             1 + self.sig_log_sq - tf.square(self.mu) - tf.exp(self.sig_log_sq),
            reduction_indices = 1
        )

        self.reconstruction_error = tf.reduce_sum(
            -self.encoder.input_data * tf.log(self.reconstruction + eps) - (1.0 - self.encoder.input_data) * tf.log(1.0 - self.reconstruction + eps),
            reduction_indices = [1,2,3] if input_channels == 3 else 0
        )
        self.error_function = tf.reduce_mean(self.vae_error + self.reconstruction_error)

    def train(self, epochs, batch_size, learning_rate, save_to=None):

        self.train_step = pt.apply_optimizer(tf.train.AdamOptimizer(learning_rate, epsilon=1), losses = [self.error_function])
        init = tf.initialize_all_variables()
        self.sess.run(init)

        for epoch in range(epochs):
            input_data = next(self.data_generator)
            _, loss_value, v = self.sess.run(
                [self.train_step, self.error_function, self.vae_error],
                {
                    self.encoder.input_data: input_data
                }
            )            

        if save_to:
            self.save(save_to)

    def generate_random_sample(self):
        pass
