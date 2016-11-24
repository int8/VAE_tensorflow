from networks import ConvolutionalEncoder, DeconvolutionalDecoder
from network_guts import decoder_tiny_sigmoid, encoder_tiny_sigmoid
from faces_reader import hdf5_generator
from vae_config import FLAGS
from vaemodel import VaeAutoencoder
from scipy.misc import imsave

if __name__ == "__main__":

    input_size = [FLAGS.input_width, FLAGS.input_height, FLAGS.input_channels]

    data_generator = hdf5_generator(FLAGS.data_file_path, FLAGS.hdf5_dataset_name, FLAGS.batch_size)
    encoder = ConvolutionalEncoder(input_size,encoder_tiny_sigmoid, FLAGS.representation_size, FLAGS.batch_size)
    decoder = DeconvolutionalDecoder(decoder_tiny_sigmoid, FLAGS.representation_size, FLAGS.batch_size)

    with  VaeAutoencoder(encoder, decoder, data_generator, FLAGS.input_channels, FLAGS.epsilon, mode = "train") as vae_autoencoder:
        vae_autoencoder.train(FLAGS.epochs, FLAGS.batch_size, FLAGS.learning_rate, "mx.tf")
        # vae_autoencoder.load("mx.tf")
        # sample = vae_autoencoder.generate_random_sample()
        # imsave('samplex.png', sample[0][0,:]  * 255)
