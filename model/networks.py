import tensorflow as tf
import numpy as np
import prettytensor as pt
from common.deconv import deconv2d

class ConvolutionalEncoder:

    def __init__(self, input_tensor_size, representation_size, batch_size):

        self.batch_size = batch_size
        self.input_tensor_size = input_tensor_size
        self.representation_size = representation_size

        self.input_data = tf.placeholder(tf.float32, [batch_size] + input_tensor_size)


    def network_guts(self):
        conv_layers = (pt.wrap(self.input_data).
                        conv2d(4, 32, stride=2, name="enc_conv1").
                        conv2d(4, 64, stride=2, name="enc_conv2").
                        conv2d(4, 128, stride=2, name="enc_conv3").
                        conv2d(4, 256, stride=2, name="enc_conv4").
                        flatten())

        mu = conv_layers.fully_connected(self.representation_size, activation_fn=None, name = "mu")
        stddev_log_sq = conv_layers.fully_connected(self.representation_size, activation_fn=None, name = "stddev_log_sq")
        return mu, stddev_log_sq

    def get_network_output(self):
        return self.network_guts();


class DeconvolutionalDecoder:

    def __init__(self, representation_size, batch_size):
        self.representation_size = representation_size
        self.batch_size = batch_size

    def network_guts(self, batch_size = None):
        batch_size = self._determine_batch_size(batch_size)

        return (pt.wrap(self.latent_var).
                fully_connected(4*256, activation_fn=None, name="dec_fc1").
                reshape([batch_size, 1, 1, 4*256]).
                deconv2d(5, 128, stride=2, edges='VALID').
                deconv2d(5, 64, stride=2, edges='VALID').
                deconv2d(6, 32, stride=2, edges='VALID').
                deconv2d(6, 3, stride=2, edges="VALID",  activation_fn=tf.nn.sigmoid, name="dec_deconv5")).tensor

    def genereate_network_output(self, input_tensor, batch_size = None):

        batch_size = self._determine_batch_size(batch_size)
        epsilon = tf.random_normal([batch_size, self.representation_size])
        self.mu = input_tensor[0]
        self.stddev = tf.sqrt(tf.exp(input_tensor[1])) # encoder output models log(sigma**2) (its domain is -inf, inf)
        self.latent_var = self.mu + epsilon * self.stddev
        return self.network_guts(), self.mu, self.stddev


    def genereate_network_output_without_noise(self, input_tensor, batch_size = None):
        self.latent_var = input_tensor[0]
        return self.network_guts()


    def generate_network_random_sample(self):
        self.epsilon = tf.placeholder(tf.float32, [1, self.representation_size])
        self.latent_var = self.epsilon
        return self.network_guts(1)

    def _determine_batch_size(self, batch_size):
        batch_size = batch_size if batch_size else self.batch_size
        return batch_size
