import scipy.io.wavfile as wavfile
import numpy as np
import tensorflow as tf
import os
from ops import *

wav_canvas_size = 2 ** 14
shuffle_data = True  # shuffle the addresses before saving
cr_train_path = './data/reindeer/cr_train/'
cr_test_path = './data/reindeer/cr_test/'
ny_train_path = './data/reindeer/p_train/'
ny_test_path = './data/reindeer/p_test/'

def slice_signal(signal, window_size, stride):
    """ Return windows of the given signal by sweeping in stride fractions
        of window
    """
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    offset = int(window_size * stride)
    slices = []
    for beg_i, end_i in zip(range(0, n_samples, offset),
                            range(window_size, n_samples + offset,
                                  offset)):
        if end_i - beg_i < window_size:
            break
        slice_ = signal[beg_i:end_i]
        if slice_.shape[0] == window_size:
            slices.append(slice_)
    return np.array(slices, dtype=np.int32)

def read_and_slice(filename, wav_canvas_size, stride):
    fm, wav_data = wavfile.read(filename)
    if fm != 16000:
        raise ValueError('Sampling rate is expected to be 16kHz!')
    signals = slice_signal(wav_data, wav_canvas_size, stride)
    return signals

def make_concat(path, s_range, n_range, stride):
    list = os.listdir(path)
    wav_list = list[s_range:n_range]
    for idx in range(len(wav_list)):
        batch_data = read_and_slice(path + wav_list[idx], wav_canvas_size, stride)
        if idx == 0:
            arr = batch_data
        else:
            arr = np.vstack((arr, batch_data))
        if idx%100 == 0:
            print(path, arr.shape, idx,'/',len(wav_list))

    row, _ = arr.shape

    return arr, row

class Loader():
    def __init__(self, batch_size=32):
        self.filenames = ["cr_train01.tfrecords", "cr_train02.tfrecords", "cr_train03.tfrecords", "cr_train04.tfrecords", "cr_train05.tfrecords",
                          "ns_train01.tfrecords", "ns_train02.tfrecords", "ns_train03.tfrecords", "ns_train04.tfrecords", "ns_train05.tfrecords"]
        self.clean = True
        self.preemph = 0.95
        found = True

        for file in self.filenames:
            if not os.path.isfile(file):
                found = False
                print('TFRECORD File is not founded')

        if not found:
            # read addresses and labels from the 'train' folder

            print("writing clean train file")
            trcr1, trcr1_row = make_concat(cr_train_path, 0, 5000, stride=0.5)
            self.create_tf_record(data=trcr1, row=trcr1_row, path='cr_train01.tfrecords')
            trcr2, trcr2_row = make_concat(cr_train_path, 5000, 10000, stride=0.5)
            self.create_tf_record(data=trcr2, row=trcr2_row, path='cr_train02.tfrecords')
            trcr3, trcr3_row = make_concat(cr_train_path, 10000, 15000, stride=0.5)
            self.create_tf_record(data=trcr3, row=trcr3_row, path='cr_train03.tfrecords')
            trcr4, trcr4_row = make_concat(cr_train_path, 15000, 20000, stride=0.5)
            self.create_tf_record(data=trcr4, row=trcr4_row, path='cr_train04.tfrecords')
            trcr5, trcr5_row = make_concat(cr_train_path, 20000, len(os.listdir(cr_train_path)), stride=0.5)
            self.create_tf_record(data=trcr5, row=trcr5_row, path='cr_train05.tfrecords')

            print("writing noisy train file")
            trns1, trns1_row = make_concat(ny_train_path, 0, 5000, stride=0.5)
            self.create_tf_record(data=trns1, row=trns1_row, path='ns_train01.tfrecords')
            trns2, trns2_row = make_concat(ny_train_path, 5000, 10000, stride=0.5)
            self.create_tf_record(data=trns2, row=trns2_row, path='ns_train02.tfrecords')
            trns3, trns3_row = make_concat(ny_train_path, 10000, 15000, stride=0.5)
            self.create_tf_record(data=trns3, row=trns3_row, path='ns_train03.tfrecords')
            trns4, trns4_row = make_concat(ny_train_path, 15000, 20000, stride=0.5)
            self.create_tf_record(data=trns4, row=trns4_row, path='ns_train04.tfrecords')
            trns5, trns5_row = make_concat(ny_train_path, 20000, len(os.listdir(ny_train_path)), stride=0.5)
            self.create_tf_record(data=trns5, row=trns5_row, path='ns_train05.tfrecords')

        self.batch_size = batch_size

    def create_tf_record(self, data, row, path):

        with tf.python_io.TFRecordWriter(path) as writer:
            for i in range(row):
                d = data[i].tostring()
                features = tf.train.Features(
                    feature = {'wav_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[d]))}
                )
                batch_data = tf.train.Example(features=features)
                serialized = batch_data.SerializeToString()
                writer.write(serialized)

    def get_dataset(self, train=True):
        if train:
            filenames_tr_cr = self.filenames[0:5]
            filenames_tr_ns = self.filenames[5:10]

        if train:
            dataset_tr_cr = tf.data.TFRecordDataset(filenames_tr_cr)
            dataset_tr_ns = tf.data.TFRecordDataset(filenames_tr_ns)

            dataset_tr_cr = dataset_tr_cr.apply(tf.contrib.data.shuffle_and_repeat(10000,seed=0))
            dataset_tr_cr = dataset_tr_cr.apply(tf.contrib.data.map_and_batch(self.parse_example,batch_size=self.batch_size,num_parallel_batches=150))
            dataset_tr_cr = dataset_tr_cr.apply(tf.contrib.data.prefetch_to_device('/device:GPU:0',100))
            iterator_tr_cr = dataset_tr_cr.make_one_shot_iterator()

            dataset_tr_ns = dataset_tr_ns.apply(tf.contrib.data.shuffle_and_repeat(10000, seed=0))
            dataset_tr_ns = dataset_tr_ns.apply(tf.contrib.data.map_and_batch(self.parse_example, batch_size=self.batch_size, num_parallel_batches=150))
            dataset_tr_ns = dataset_tr_ns.apply(tf.contrib.data.prefetch_to_device('/device:GPU:0', 100))
            iterator_tr_ns = dataset_tr_ns.make_one_shot_iterator()

            return iterator_tr_cr, iterator_tr_ns

    def parse_example(self, serialized):

        features = tf.parse_single_example(serialized=serialized, features={
            'wav_raw': (tf.FixedLenFeature((), tf.string, default_value=""))})

        wav_data = features['wav_raw']
        wave = tf.decode_raw(wav_data, tf.int32)

        wave.set_shape(wav_canvas_size)
        wave = (2. / 65535.) * tf.cast((wave - 32767), tf.float32) + 1.

        if self.preemph > 0:
            wave = tf.cast(pre_emph(wave, self.preemph), tf.float32)
        return tf.reshape(wave, [16384, 1])
