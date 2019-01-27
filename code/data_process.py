from make_tfrecord import Loader
import tensorflow as tf
from scipy.io import wavfile
import numpy as np
from ops import pre_emph
import os
from segan import SEGAN

loader = Loader()
cr, ns = loader.get_dataset()

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
    return np.array(slices, dtype=np.float32)

def pre_emph_test(coeff, canvas_size):

    x_ = tf.placeholder(tf.float32, shape=[canvas_size,])
    x_preemph = pre_emph(x_, coeff)
    return x_, x_preemph


with tf.Session() as sess:
    fm, wav_data = wavfile.read('./test/test.wav')

    wave = (2. / 65535.) * (wav_data.astype(np.float32) - 32767) + 1.
    x_pholder, preemph_op = pre_emph_test(0.95, wave.shape[0])
    print(preemph_op)
    wave = sess.run(preemph_op, feed_dict={x_pholder: wave})
    print(wave)

    se_model = SEGAN(sess)
    sam_wav = slice_signal(wave, 16384, 1)

    sam_wav = tf.convert_to_tensor(sam_wav)
    print(sess.run(sam_wav))
    c_wave = se_model.generator(sam_wav)

    t_wave_result = tf.reshape(c_wave, [c_wave.get_shape().as_list()[0]*c_wave.get_shape().as_list()[1]])

    init = tf.global_variables_initializer()
    sess.run(init)
    array = sess.run(t_wave_result)

    #wavfile.write(os.path.join('aa.wav'), 16000, array)
