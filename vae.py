import numpy as np
import tensorflow as tf
from scipy.misc import imsave
from common.vae_args import args
from common.io_utils import HDF5Reader, read_data_from_dir
from model.network_implementations import ConvolutionalEncoder, DeconvolutionalDecoder
from model.vaemodel import VaeAutoencoderTrainer, VaeAutoencoderSampler, VaeAutoencoderReconstructor

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

    elif args.command == "reconstruct":

        input_size = [args.input_width, args.input_height, 3]
        # data from directory
        data = read_data_from_dir(args.inputdir, input_size)
        # build encoder graph
        encoder = ConvolutionalEncoder(input_size, args.latent_dim , len(data['files']))
        # build decoder graph
        decoder = DeconvolutionalDecoder(args.latent_dim, len(data['files']))

        with VaeAutoencoderReconstructor(encoder, decoder, args.input) as reconstructor:
            r = reconstructor.reconstruct(data['tensors'])
            for i in xrange(r.shape[0]):
                imsave(args.output + '/reconstruction_' + str(data['files'][i]), r[i,:,:,:])

    elif args.command == "sample":
        # build decoder graph
        decoder = DeconvolutionalDecoder(args.latent_dim, 1)
        with VaeAutoencoderSampler(decoder, args.input) as sampler:
            a =  np.random.randn(1,args.latent_dim)
            b =  np.random.randn(1,args.latent_dim)

            for k in xrange(30):
                a = b.copy()
                b = np.random.randn(1,args.latent_dim)
                samples = sampler.walk_between_points(a, b, 10)
                for i in xrange(len(samples)):
                    imsave(args.output_dir + str(k) + '_' + str(i) + '.png', samples[i][0,:,:,:])
