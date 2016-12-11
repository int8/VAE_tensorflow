import tensorflow as tf
import numpy as np
import prettytensor as pt
from common.deconv import deconv2d
from model.network_base import VAEEncoderBase, VAEDecoderBase

class ConvolutionalEncoder(VAEEncoderBase):

    def __init__(self, input_tensor_size, representation_size, batch_size):
        VAEEncoderBase.__init__(self, input_tensor_size, representation_size, batch_size)

    def guts(self):
        conv_layers = (pt.wrap(self.input_data).
                        conv2d(4, 32, stride=2, name="enc_conv1").
                        conv2d(4, 64, stride=2, name="enc_conv2").
                        conv2d(4, 128, stride=2, name="enc_conv3").
                        conv2d(4, 256, stride=2, name="enc_conv4").
                        flatten())

        mu = conv_layers.fully_connected(self.representation_size, activation_fn=None, name = "mu")
        stddev_log_sq = conv_layers.fully_connected(self.representation_size, activation_fn=None, name = "stddev_log_sq")
        return mu, stddev_log_sq


class DeconvolutionalDecoder(VAEDecoderBase):

    def __init__(self, representation_size, batch_size):
        VAEDecoderBase.__init__(self, representation_size, batch_size)

    def guts(self, batch_size = None):
        batch_size = self._determine_batch_size(batch_size)

        return (pt.wrap(self.latent_var).
                fully_connected(4*256, activation_fn=None, name="dec_fc1").
                reshape([batch_size, 1, 1, 4*256]).
                deconv2d(5, 128, stride=2, edges='VALID', name="dec_deconv2").
                deconv2d(5, 64, stride=2, edges='VALID', name="dec_deconv3").
                deconv2d(6, 32, stride=2, edges='VALID', name="dec_deconv4").
                deconv2d(6, 3, stride=2, edges="VALID",  activation_fn=tf.nn.sigmoid, name="dec_deconv5")).tensor
