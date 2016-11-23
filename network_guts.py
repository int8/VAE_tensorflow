from deconv import deconv2d
import prettytensor as pt
import tensorflow as tf

def encoder_tiny_sigmoid(representation_size, input):
    return (pt.wrap(input).
                conv2d(5, 5, stride=2).
                conv2d(5, 5, stride=2).
                conv2d(5, 5).
                conv2d(5, 5).
                flatten().
                fully_connected(representation_size * 2, activation_fn=tf.nn.sigmoid)).tensor

def decoder_tiny_sigmoid(batch_size, representation_size, input = None):
    epsilon = tf.random_normal([batch_size, representation_size])
    if input is None:
        mean = None
        stddev = None
        input = epsilon
    else:
        mean = input[:, :representation_size]
        stddev = tf.sqrt(tf.exp(input[:, representation_size:]))
        input = mean + epsilon * stddev
    return (pt.wrap(input).
            reshape([batch_size, 1, 1, representation_size]).
            deconv2d(8, 5, edges='VALID').
            deconv2d(8, 5, stride=2, edges='VALID').
            deconv2d(8, 5, stride=2, edges='VALID').
            deconv2d(8, 5, stride=2, edges="VALID").
            deconv2d(7, 3, edges="VALID",  activation_fn=tf.nn.sigmoid).tensor, mean, stddev)
