import h5py
import numpy as np
import os
import Image
import PIL
from scipy.misc import imread
from progressbar import ProgressBar, Percentage, Bar

class HDF5Reader:
    def __init__(self, filename,  dataset_name, b_size):
        with h5py.File(filename,'r') as hf:
            self.data = np.array(hf.get(dataset_name))
        self.i = 0
        self.b_size = b_size
        self.n = self.data.shape[0]
        self.cycles = 0

    def next(self):
        k_left = self.i
        k_right = self.i + self.b_size
        if k_right >= self.n:
            batch = np.append(self.data[range(k_left, self.n),:,:,:], self.data[range(0, k_right - self.n),:,:,:], axis = 0).astype(np.float32) / 255.
            self.data = self.data[np.random.permutation(self.n),:]
            self.cycles = self.cycles + 1
        else:
            batch = self.data[k_left:k_right, :, :, :].astype(np.float32) / 255.
        self.i = k_right % self.n
        return batch

    def get_cycles(self):
        return self.cycles


def hdf5_generator(filename, dataset_name, batch_size):

    with h5py.File(filename,'r') as hf:
        data = np.array(hf.get(dataset_name))
    i = 0
    nr_of_points = data.shape[0]
    while True:
        key_left = i
        key_right = i + batch_size
        if key_right >= nr_of_points:
            yield np.append(data[range(key_left, nr_of_points),:,:,:], data[range(0, key_right - nr_of_points),:,:,:], axis = 0).astype(np.float32) / 255.
            data = data[np.random.permutation(data.shape[0]),:]
        else:
            yield data[key_left:key_right, :, :, :].astype(np.float32) / 255.
        i = key_right % nr_of_points

def resize_hdf5(input_filename, output_filename, dataset_name, batch_size, new_width = 64, new_height = 64):

    with h5py.File(filename,'r') as hf:
        data = np.array(hf.get(dataset_name))
    nr_of_points = data.shape[0]
    depth = data.shape[3]
    newdata = np.empty(shape=(nr_of_points, new_width, new_height, depth), dtype=np.uint8)

    for i in xrange(nr_of_points):
        datum = data[i,:,:,:]
        resized_datum = np.asarray(Image.fromarray((datum * 255).astype(np.uint8), 'RGB').resize((new_width,new_height), PIL.Image.ANTIALIAS))
        newdata[i,:,:,:] = resized_datum

    with h5py.File(output_filename, 'w') as hf:
        hf.create_dataset(dataset_name, data = newdata)


def images_to_hdf5(dir_path, output_hdf5, size = (112,112), channels = 3, resize_to = None):
    files = sorted(os.listdir(dir_path))
    nr_of_images = len(files)
    if resize_to:
        size = resize_to
    i = 0
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=nr_of_images).start()
    data = np.empty(shape=(nr_of_images, size[0], size[1], channels), dtype=np.uint8)
    for f in files:
        datum = imread(dir_path + '/' + f)
        if resize_to:
            datum = np.asarray(Image.fromarray((datum), 'RGB').resize((size[0],size[1]), PIL.Image.ANTIALIAS))
        data[i,:,:,:] = datum
        i = i + 1
        pbar.update(i)
    pbar.finish()
    with h5py.File(output_hdf5, 'w') as hf:
        hf.create_dataset('data', data=data)

def read_data_from_dir(dir_path, resize_to):
    files = os.listdir(dir_path)
    nr_of_images = len(files)
    data = {
        'files': ['' for _ in xrange(nr_of_images)],
        'tensors': np.empty(shape=(nr_of_images, resize_to[0], resize_to[1], resize_to[2]), dtype=np.uint8)
    }
    i = 0
    for f in files:
        datum = imread(dir_path + '/' + f)
        datum = np.asarray(Image.fromarray((datum), 'RGB').resize((resize_to[0],resize_to[1]), PIL.Image.ANTIALIAS))
        data['tensors'][i, :, :, :] = datum
        data['files'][i] = f
        i = i + 1
    return data
