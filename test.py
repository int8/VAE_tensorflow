from model.networks import ConvolutionalEncoder, DeconvolutionalDecoder
from common.io_utils import images_to_hdf5, HDF5Reader
from model.vaemodel import VaeAutoencoderTrainer, VaeAutoencoderSampler
from scipy.misc import imsave
import numpy as np
import tensorflow as tf

if __name__ == "__main__":

    images_to_hdf5('../VAE-pretty-tensor/datasets/TrumpImagesReady/thedonald' , 'data/doland2_6464.h5', resize_to = (64,64))
    # reader = HDF5Reader(filename = 'hilary2_6464.h5', dataset_name='data', b_size=1)
    # elem = reader.next()
    # imsave('hil.png', elem[0,:,:,:])
