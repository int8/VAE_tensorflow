from networks import ConvolutionalEncoder, DeconvolutionalDecoder
from network_guts import decoder_tiny_sigmoid, encoder_tiny_sigmoid
from faces_reader import hdf5_generator
from vae_config import FLAGS
from vaemodel import VaeAutoencoder

if __name__ == "__main__":

    input_size = [FLAGS.input_width, FLAGS.input_height, FLAGS.input_channels]

    data_generator = hdf5_generator(FLAGS.data_file_path, FLAGS.hdf5_dataset_name, FLAGS.batch_size)
    encoder = ConvolutionalEncoder(input_size,encoder_tiny_sigmoid, FLAGS.representation_size, FLAGS.batch_size)
    decoder = DeconvolutionalDecoder(decoder_tiny_sigmoid, FLAGS.representation_size, FLAGS.batch_size)
    vae_autoencoder = VaeAutoencoder(encoder, decoder, data_generator, FLAGS.input_channels, FLAGS.epsilon)
    vae_autoencoder.train(FLAGS.epochs, FLAGS.batch_size, FLAGS.learning_rate)
