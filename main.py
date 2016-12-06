from networks import ConvolutionalEncoder, DeconvolutionalDecoder
from io_utils import hdf5_generator
from vae_config import args
from vaemodel import VaeAutoencoder
from scipy.misc import imsave
import numpy as np

if __name__ == "__main__":

    input_size = [args.input_width, args.input_height, 3]
    data_generator = hdf5_generator(args.input, args.hdf5_dataset_name, args.batch_size)
    encoder = ConvolutionalEncoder(input_size, args.latent_dim, args.batch_size)
    decoder = DeconvolutionalDecoder(args.latent_dim, args.batch_size)

    with  VaeAutoencoder(encoder, decoder, data_generator, mode = "train") as vae_autoencoder:
        vae_autoencoder.train(args.epochs, args.batch_size, args.learning_rate)
        # vae_autoencoder.load("mx.tf")
