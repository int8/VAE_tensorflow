# VAE implementation in Tensorflow for face expression reconstruction

This is supplementary code for the blog post [here](http://int8.io/variational-autoencoder-in-tensorflow/)

The main motivation of this work is to use Variational Autoencoder model to embed unseen faces into the latent space of pre-trained single actor-centric face expressions data. Three datasets are used in described experiments. They are based on youtube videos passed through openface feature extraction utility


short demo video available [here](http://int8.io/wp-content/uploads/2016/12/output.mp4)


## Requirements:

* python v2.7.6
* numpy v1.11.1
* scipy v0.13.3
* h5py v2.6.0
* Pillow v2.3.0
* progressbar v2.3
* argparse v1.2.1
* tensorflow 0.9.0
* prettytensor 0.6.2


for help try
```bash
$ python vae.py -h

usage: vae.py [-h] {train,sample,reconstruct} ...

positional arguments:
  {train,sample,reconstruct}
    train               train VAE model [vae.py train -h]
    sample              sample from existing model [vae.py sample -h]
    reconstruct         reconstruct images based on existing model [vae.py
                        reconstruct -h]

optional arguments:
  -h, --help            show this help message and exit
```

The tool implements 3 high level commands:

* train

```bash
$ python vae.py train -h
vagrant@vagrant-ubuntu-trusty-64:~/tflow/VAE_OOP$ python vae.py train -h
usage: vae.py train [-h] [--hdf5-dataset-name] [--batch-size] [--epochs]
                    [--learning-rate] [--latent-dim] [--input-width]
                    [--input-height]
                    input output

positional arguments:
  input                 Path of HDF5 data file
  output                Output tf model dir

optional arguments:
  -h, --help            show this help message and exit
  --hdf5-dataset-name   Name of dataset in hdf5
  --batch-size          Batch size
  --epochs              Number of epochs to run
  --learning-rate       Learning rate
  --latent-dim          latent variable dimensionality
  --input-width         Width of input images
  --input-height        Height of input images
```

the most important detail here is format of input data. Input is expected to be HDF5 file containing dataset that is 4 dimensional tensor of size:
```
nr_of_objects x width x height x 3
```

the output directory by default will be populated with tensorflow saved model + metadata files




* sample

when your model is fully trained you can use sample command to draw points from latent space and walk from one to another randomly

```bash
$ python vae.py sample -h
usage: vae.py sample [-h] [--latent-dim] [--input-width] [--input-height]
                     [--output--dir]
                     input

positional arguments:
  input            Path of tensorflow model dir

optional arguments:
  -h, --help       show this help message and exit
  --latent-dim     latent variable dimensionality
  --input-width    Width of input images
  --input-height   Height of input images
  --output--dir    Output dir where png files are stored
```

* reconstruct

If you want to use existing model to reconstruct input images use reconstruct command

```bash
$ python vae.py reconstruct -h
usage: vae.py reconstruct [-h] [--latent-dim] [--input-width] [--input-height]
                          input inputdir output

positional arguments:
  input            Path of tensorflow model dir
  inputdir         Path directory where input images are stored
  output           Output directory where reconstructions will be stored

optional arguments:
  -h, --help       show this help message and exit
  --latent-dim     latent variable dimensionality
  --input-width    Width of model input images
  --input-height   Height of model input images

```
