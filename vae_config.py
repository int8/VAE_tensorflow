import tensorflow as tf
import prettytensor as pt

flags = tf.flags
flags.DEFINE_integer("batch_size", 10,  "batch size")
flags.DEFINE_integer("epochs", 1000, "numbers of epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_integer("representation_size", 5, "size of the VAE representaton")
flags.DEFINE_float("epsilon", 1e-8, "tiny value to avoid nans across network")
flags.DEFINE_integer("input_channels", 3, "number of input channels")
flags.DEFINE_integer("input_width", 112, "input width")
flags.DEFINE_integer("input_height",112, "input height")
flags.DEFINE_string("data_file_path","../VAE-pretty-tensor/datasets/hdf5/donald1.hdf5", "Path of the hdf5 data file")
flags.DEFINE_string("hdf5_dataset_name","data", "Name of dataset in hdf5")

FLAGS = flags.FLAGS

pt_default_scope = pt.defaults_scope(activation_fn=tf.nn.sigmoid,
                        batch_normalize=True,
                        learned_moments_update_rate=0.0003,
                        variance_epsilon=0.001,
                        scale_after_normalization=True)
