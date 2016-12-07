from model.networks import ConvolutionalEncoder, DeconvolutionalDecoder
from common.io_utils import HDF5Reader
from common.vae_args import args
from model.vaemodel import VaeAutoencoderTrainer, VaeAutoencoderSampler
from scipy.misc import imsave
import numpy as np
import tensorflow as tf

if __name__ == "__main__":

    if args.command == "train":
        input_size = [args.input_width, args.input_height, 3]
        hdf5reader = HDF5Reader(args.input, args.hdf5_dataset_name, args.batch_size)
        # build encoder graph
        encoder = ConvolutionalEncoder(input_size, args.latent_dim, args.batch_size)
        # build decoder graph
        decoder = DeconvolutionalDecoder(args.latent_dim, args.batch_size)

        with VaeAutoencoderTrainer(encoder, decoder, hdf5reader) as trainer:
            trainer.train(args.epochs, args.batch_size, args.learning_rate)
            trainer.save(args.output)

    elif args.command == "sample":
        # build encoder graph
        decoder = DeconvolutionalDecoder(args.latent_dim, 1)
        with VaeAutoencoderSampler(decoder, args.input) as sampler:
            a = np.random.rand(1,30)
            b = -np.random.rand(1,30)
            samples = sampler.walk_between_points(a, b, 100)
            for i in xrange(len(samples)):
                imsave('samples/' + str(i) + '.png', samples[i][0,:,:,:])
