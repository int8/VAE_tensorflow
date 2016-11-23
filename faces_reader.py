import h5py
import numpy as np

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
        else:
            yield data[key_left:key_right, :, :, :].astype(np.float32) / 255.
        i = key_right % nr_of_points
