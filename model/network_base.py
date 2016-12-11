import tensorflow as tf

class VAEEncoderBase:

    def __init__(self, input_tensor_size, representation_size, batch_size):

        self.batch_size = batch_size
        self.input_tensor_size = input_tensor_size
        self.representation_size = representation_size

        self.input_data = tf.placeholder(tf.float32, [batch_size] + input_tensor_size)

    def guts(self, batch_size = None):
        raise("Implement guts() function in your Encoder Implementation")

    def get_network_output(self):
        return self.guts();


class VAEDecoderBase:

    def __init__(self, representation_size, batch_size):
        self.representation_size = representation_size
        self.batch_size = batch_size

    def guts(self, batch_size = None):
        raise("Implement guts() function in your Decoder Implementation")

    def genereate_network_output(self, input_tensor, batch_size = None):

        batch_size = self._determine_batch_size(batch_size)
        epsilon = tf.random_normal([batch_size, self.representation_size])
        self.mu = input_tensor[0]
        # encoder output is suposed ty model log(sigma**2) (its domain is -inf, inf)
        self.stddev = tf.sqrt(tf.exp(input_tensor[1]))
        self.latent_var = self.mu + epsilon * self.stddev
        return self.guts(), self.mu, self.stddev


    def genereate_network_output_without_noise(self, input_tensor, batch_size = None):
        self.latent_var = input_tensor[0]
        return self.guts()


    def generate_network_random_sample(self):
        self.epsilon = tf.placeholder(tf.float32, [1, self.representation_size])
        self.latent_var = self.epsilon
        return self.guts(1)

    def _determine_batch_size(self, batch_size):
        batch_size = batch_size if batch_size else self.batch_size
        return batch_size
