import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from bnorm import *
import numpy as np

def downconv(x, output_dim, kwidth=5, pool=2, init=None, uniform=False,
             bias_init=None,name='downconv'):
    """ Downsampled convolution 1d """
    x2d = tf.expand_dims(x, 2)
    with tf.variable_scope(name):
        W = tf.get_variable('W', [kwidth, 1, x.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(x2d, W, strides=[1, pool, 1, 1], padding='SAME')

        if bias_init is not None:
            b = tf.get_variable('b', [output_dim], initializer=bias_init)
            conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
        else:
            conv = tf.reshape(conv, conv.get_shape())

        conv = tf.reshape(conv, conv.get_shape().as_list()[:2] +
                          [conv.get_shape().as_list()[-1]])
        return conv

def prelu(x, name='prelu'):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        # make one alpha per feature
        alpha = tf.get_variable('alpha', in_shape[-1],
                                initializer=tf.constant_initializer(0.),
                                dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alpha * (x - tf.abs(x)) * .5

        # return ref to alpha vector
        return pos + neg, alpha

def make_z(shape):
    z = tf.random_normal(shape, mean=0., stddev=1., name='z', dtype=tf.float32)
    return z

def deconv(x, output_shape, kwidth=5, dilation=2, init=None, uniform=False,
           bias_init=None, name='deconv1d'):
    input_shape = x.get_shape()
    in_channels = input_shape[-1]
    out_channels = output_shape[-1]
    assert len(input_shape) >= 3
    # reshape the tensor to use 2d operators
    x2d = tf.expand_dims(x, 2)
    o2d = output_shape[:2] + [1] + [output_shape[-1]]

    with tf.variable_scope(name):
        # filter shape: [kwidth, output_channels, in_channels]
        W = tf.get_variable('W', [kwidth, 1, out_channels, in_channels],
                            initializer=tf.contrib.layers.xavier_initializer())

        deconv = tf.nn.conv2d_transpose(x2d, W, output_shape=o2d,
                                            strides=[1, dilation, 1, 1])
        if bias_init is not None:
            b = tf.get_variable('b', [out_channels],initializer=tf.constant_initializer(0.))
            deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
        else:
            deconv = tf.reshape(deconv, deconv.get_shape())
        # reshape back to 1d
        deconv = tf.reshape(deconv, output_shape)
        return deconv

def conv1d(x, kwidth=5, num_kernels=1, init=None, uniform=False, bias_init=None,
           name='conv1d', padding='SAME'):
    input_shape = x.get_shape()
    in_channels = input_shape[-1]
    assert len(input_shape) >= 3

    with tf.variable_scope(name):
        # filter shape: [kwidth, in_channels, num_kernels]
        W = tf.get_variable('W', [kwidth, in_channels, num_kernels],
                            initializer=tf.contrib.layers.xavier_initializer()
                            )
        conv = tf.nn.conv1d(x, W, stride=1, padding=padding)
        if bias_init is not None:
            b = tf.get_variable('b', [num_kernels],
                                initializer=tf.constant_initializer(bias_init))
            conv = conv + b
        return conv

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=input_layer.get_shape().as_list(),
                             mean=0.0,
                             stddev=std,
                             dtype=tf.float32)
    return input_layer + noise

def leakyrelu(x, alpha=0.3, name='lrelu'):
    return tf.maximum(x, alpha * x, name=name)

def vbn(tensor, name):
    VBN_cls = VBN
    vbn = VBN_cls(tensor, name)
    return vbn(tensor)

def pre_emph(x, coeff=0.95):

    x0 = tf.reshape(x[0], [1,])
    diff = x[1:] - coeff * x[:-1]
    concat = tf.concat([x0, diff], axis=0)

    return concat

def de_emph(y, coeff=0.95):

    if coeff <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coeff * x[n - 1] + y[n]
    return x

def pre_emph_test(coeff, canvas_size):

    x_ = tf.placeholder(tf.float32, shape=[canvas_size,])
    x_preemph = pre_emph(x_, coeff)
    return x_, x_preemph

def signal_slice(signal, window_size, stride):
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


def discriminator_loss(real, fake):

    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    loss = real_loss + fake_loss

    return loss


def generator_loss(fake):

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    return loss

