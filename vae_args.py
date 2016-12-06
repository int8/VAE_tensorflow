import argparse

parser = argparse.ArgumentParser(prog='main.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', help='Path of HDF5 data file')
parser.add_argument('--hdf5-dataset-name', help='Name of dataset in hdf5', default = 'data', metavar='')

parser.add_argument('--batch-size', help='Batch size', default = 128, type=int, metavar='')
parser.add_argument('--epochs', help='Number of epochs to run', default = 100, type=int, metavar='')
parser.add_argument('--learning-rate', help='Learning rate', default = 1e-2, type=float, metavar='')
parser.add_argument('--latent-dim', help='latent variable dimensionality', default = 30, type=int, metavar='')
parser.add_argument('--input-width', help='Width of input images', default = 64, type=int, metavar='')
parser.add_argument('--input-height', help='Height of input images', default = 64, type=int, metavar='')

args = parser.parse_args()
