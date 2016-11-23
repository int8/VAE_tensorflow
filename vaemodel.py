import tensorflow as tf
import prettytensor as pt
from vae_config import pt_default_scope

class VaeAutoencoder:

    def __init__(self, encoder, decoder, data_generator, input_channels, eps):

        self.encoder = encoder
        self.decoder = decoder
        self.data_generator = data_generator
        with pt_default_scope:
            self.reconstruction, self.mean, self.stddev = self.decoder.get_network_output(self.encoder.get_network_output())

        self.build_error_function(input_channels, eps)

    def build_error_function(self, input_channels, eps):
        self.vae_error = tf.reduce_sum(
            0.5 * (tf.square(self.mean) + tf.square(self.stddev) - 2.0 * tf.log(tf.clip_by_value(self.stddev, eps, 1.0)) - 1.0),
            reduction_indices = 1
        )
        self.reconstruction_error = tf.reduce_sum(
            -self.encoder.input_data * tf.log(self.reconstruction + eps) - (1.0 - self.encoder.input_data) * tf.log(1.0 - self.reconstruction + eps),
            reduction_indices = [1,2,3] if input_channels == 3 else 0
        )
        self.error_function = tf.reduce_mean(self.vae_error + self.reconstruction_error)

    def train(self, epochs, batch_size, learning_rate):

        self.train_step = pt.apply_optimizer(tf.train.AdamOptimizer(learning_rate, epsilon=1), losses = [self.error_function])
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(epochs):
                input_data = next(self.data_generator)
                _, loss_value = sess.run([self.train_step, self.error_function], {self.encoder.input_data: input_data })
                print(loss_value)
