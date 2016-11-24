import tensorflow as tf

class ConvolutionalEncoder:

    def __init__(self, input_tensor_size, network_guts_func, representation_size, batch_size):

        self.batch_size = batch_size
        self.input_tensor_size = input_tensor_size
        self.representation_size = representation_size

        self.input_data = tf.placeholder(tf.float32, [batch_size] + input_tensor_size)
        self.network_guts_func = network_guts_func


    def get_network_output(self):
        return self.network_guts_func(self.representation_size, self.input_data);


class DeconvolutionalDecoder:

    def __init__(self, network_guts_func, representation_size, batch_size):

        self.network_guts_func = network_guts_func
        self.representation_size = representation_size
        self.batch_size = batch_size
        self.graph_built = False

    def get_network_output(self, representation, batch_size = None):
        batch_size = batch_size if batch_size else self.batch_size
        return self.network_guts_func(batch_size, self.representation_size, representation);
