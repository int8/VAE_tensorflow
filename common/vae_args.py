import argparse

parser = argparse.ArgumentParser(prog='vae.py')

subparsers = parser.add_subparsers(dest="command")

train_parser = subparsers.add_parser("train", help="train VAE model [vae.py train -h]")
sample_parser = subparsers.add_parser("sample", help="sample from existing model [vae.py sample -h]")
recon_parser = subparsers.add_parser("reconstruct", help="reconstruct images based on  existing model [vae.py reconstruct -h]")

train_parser.add_argument('input', help='Path of HDF5 data file')
train_parser.add_argument('output', help='Output tf model dir', type=str)
train_parser.add_argument('--hdf5-dataset-name', help='Name of dataset in hdf5', default = 'data', metavar='')
train_parser.add_argument('--batch-size', help='Batch size', default = 128, type=int, metavar='')
train_parser.add_argument('--epochs', help='Number of epochs to run', default = 100, type=int, metavar='')
train_parser.add_argument('--learning-rate', help='Learning rate', default = 1e-2, type=float, metavar='')
train_parser.add_argument('--latent-dim', help='latent variable dimensionality', default = 30, type=int, metavar='')
train_parser.add_argument('--input-width', help='Width of input images', default = 64, type=int, metavar='')
train_parser.add_argument('--input-height', help='Height of input images', default = 64, type=int, metavar='')

sample_parser.add_argument('input', help='Path of tensorflow model dir', type=str)
sample_parser.add_argument('--latent-dim', help='latent variable dimensionality', default = 30, type=int, metavar='')
sample_parser.add_argument('--input-width', help='Width of input images', default = 64, type=int, metavar='')
sample_parser.add_argument('--input-height', help='Height of input images', default = 64, type=int, metavar='')
sample_parser.add_argument('--output--dir', help='Output dir where png files are stored ', type=str, default='samples/', metavar='')

recon_parser.add_argument('input', help='Path of tensorflow model dir', type=str)
recon_parser.add_argument('inputdir', help='Path directory where input images are stored', type=str)
recon_parser.add_argument('output', help='Output directory where reconstructions will be stored', type=str)
recon_parser.add_argument('--latent-dim', help='latent variable dimensionality', default = 30, type=int, metavar='')
recon_parser.add_argument('--input-width', help='Width of model input images', default = 64, type=int, metavar='')
recon_parser.add_argument('--input-height', help='Height of model input images', default = 64, type=int, metavar='')


args = parser.parse_args()
