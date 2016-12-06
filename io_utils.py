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
            data = data[np.random.permutation(data.shape[0]),:]
        else:
            yield data[key_left:key_right, :, :, :].astype(np.float32) / 255.
        i = key_right % nr_of_points

def resize_hdf5(input_filename, output_filename, dataset_name, batch_size, new_width = 64, new_height = 64):

    with h5py.File(filename,'r') as hf:
        data = np.array(hf.get(dataset_name))
    nr_of_points = data.shape[0]
    depth = data.shape[3]
    newdata = np.empty(shape=(nr_of_points, new_width, new_height, depth), dtype=np.byte)

    for i in xrange(nr_of_points):
        datum = data[i,:,:,:]
        resized_datum = np.asarray(Image.fromarray((datum * 255).astype(np.byte), 'RGB').resize((new_width,new_height), PIL.Image.ANTIALIAS))
        newdata[i,:,:,:] = resized_datum

    with h5py.File(output_filename, 'w') as hf:
        hf.create_dataset(dataset_name, data = newdata)
