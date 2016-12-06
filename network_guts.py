from deconv import deconv2d
import prettytensor as pt
import tensorflow as tf

def encoder_tiny_sigmoid(representation_size, input):
    layer =  (pt.wrap(input).
                conv2d(5, 5, stride=2, name="enc_conv1").
                conv2d(5, 5, stride=2, name="enc_conv2").
                conv2d(5, 5, name="enc_conv3").
                conv2d(5, 5, name="enc_conv4").
                flatten())

    mu = layer.fully_connected(representation_size, activation_fn=None, name = "enc_fc1")
    sig_log_sq = layer.fully_connected(representation_size, activation_fn=None, name = "enc_fc2")
    return mu, sig_log_sq

def generate_decoder_input(batch_size, representation_size, representation):
    
    return input, representation[0], stddev

def decoder_tiny_sigmoid(batch_size, representation_size, representation):
    input, mu, sig_log_sq = generate_decoder_input(batch_size, representation_size, representation)
    return ((pt.wrap(input).
            reshape([batch_size, 1, 1, representation_size]).
            deconv2d(8, 5, edges='VALID', name="dec_deconv1").
            deconv2d(8, 5, stride=2, edges='VALID', name="dec_deconv2").
            deconv2d(8, 5, stride=2, edges='VALID', name="dec_deconv3").
            deconv2d(8, 5, stride=2, edges="VALID", name="dec_deconv4").
            deconv2d(7, 3, edges="VALID",  activation_fn=tf.nn.sigmoid, name="dec_deconv5")).tensor, mu, sig_log_sq, input)

def decoder
