import tensorflow as tf
import numpy as np
from ops import *
from make_tfrecord import Loader

def discriminator(wave_in):
    #d_num_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
    d_num_fmaps = [32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 2048]
    #wave_in: waveform input
    # take the waveform as input "activation"

    in_dims = wave_in.get_shape().as_list()
    hi = wave_in
    if len(in_dims) == 2:
        hi = tf.expand_dims(wave_in, -1)
    elif len(in_dims) < 2 or len(in_dims) > 3:
        raise ValueError('Input error')

    batch_size = int(wave_in.get_shape()[0])

    # set up the disc_block function

    with tf.variable_scope('d_model') as scope:
        def disc_block(block_idx, input_, kwidth, nfmaps, bnorm, activation, pooling=2):
            with tf.variable_scope('d_block_{}'.format(block_idx),reuse=tf.AUTO_REUSE):
                bias_init = tf.constant_initializer(0.)
                hi_a = downconv(input_, nfmaps, kwidth=kwidth, pool=pooling,
                                init=tf.truncated_normal_initializer(stddev=0.02),
                                bias_init=bias_init
                                )
                hi_a = vbn(hi_a, 'd_vbn_{}'.format(block_idx))
                hi = leakyrelu(hi_a, name='lrelu')

                return hi

        # apply input noisy layer to real and fake samples

        hi = gaussian_noise_layer(hi, 0.)

        for block_idx, fmaps in enumerate(d_num_fmaps):
            hi = disc_block(block_idx, hi, 31,
                            d_num_fmaps[block_idx],
                            True, 'leakyrelu')
            print(block_idx, '번째 out =', hi.get_shape())
        # hi_f = flatten(hi)

        # hi_f = tf.nn.dropout(hi_f, self.keep_prob_var)

        d_logit_out = conv1d(hi, kwidth=1, num_kernels=1,
                             init=tf.truncated_normal_initializer(stddev=0.02),
                             name='logits_conv')
        print(d_logit_out.get_shape())
        d_logit_out = tf.squeeze(d_logit_out)
        d_logit_out = fully_connected(d_logit_out, 1, activation_fn=None)
        print('discriminator output shape: ', d_logit_out.get_shape())
        print('*****************************')
        return d_logit_out


def generator(noisy_w):
    kwidth = 31
    alphas = []
    skips = []
    in_dims = noisy_w.get_shape().as_list()
    h_i = noisy_w
    print(h_i.get_shape())
    batch_size = int(noisy_w.get_shape()[0])
    if len(in_dims) == 2:
        h_i = tf.expand_dims(noisy_w, -1)
    elif len(in_dims) < 2 or len(in_dims) > 3:
        raise ValueError('Input error')
    with tf.variable_scope('g_model'):
        # FIRST ENCODER
        # enc ~ [16384x1, 8192x16, 4096x32, 2048x32, 1024x64, 512x64, 256x128, 128x128, 64x256, 32x256, 16x512, 8x1024]
        # dec ~ [8x2048, 16x1024, 32x512, 64x512, 8x256, 256x256, 512x128, 1024x128, 2048x64, 4096x64, 8192x32, 16384x1]
        g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        bias_init = tf.constant_initializer(0.)
        for layer_idx, layer_depth in enumerate(g_enc_depths):
            h_i_dwn = downconv(h_i, layer_depth, kwidth=kwidth,
                               init=tf.truncated_normal_initializer(stddev=0.02), bias_init=bias_init,
                               name='enc_{}'.format(layer_idx))
            h_i = h_i_dwn
            if layer_idx < len(g_enc_depths) - 1:
                # store skip connection
                # last one is not stored cause it's the code
                skips.append(h_i)
            h_i = prelu(h_i, name='enc_prelu_{}'.format(layer_idx))
            # split h_i into its components
            alpha_i = h_i[1]
            h_i = h_i[0]
            alphas.append(alpha_i)
        print("중간", h_i.get_shape())
        # concat c & z
        z = make_z([batch_size, h_i.get_shape().as_list()[1],
                    g_enc_depths[-1]])
        h_i = tf.concat([z, h_i], -1)

        # SECOND DECODER (reverse order)
        g_dec_depths = [512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 1]
        for layer_idx, layer_depth in enumerate(g_dec_depths):
            h_i_dim = h_i.get_shape().as_list()
            out_shape = [h_i_dim[0], h_i_dim[1] * 2, layer_depth]
            bias_init = tf.constant_initializer(0.)
            # deconv
            h_i_dcv = deconv(h_i, out_shape, kwidth=kwidth, dilation=2,
                             init=tf.truncated_normal_initializer(stddev=0.02),
                             bias_init=bias_init,
                             name='dec_{}'.format(layer_idx))
            h_i = h_i_dcv

            if layer_idx < len(g_dec_depths) - 1:
                h_i = prelu(h_i, name='dec_prelu_{}'.format(layer_idx))
                alpha_i = h_i[1]
                h_i = h_i[0]
                alphas.append(alpha_i)
                skip_ = skips[-(layer_idx + 1)]
                # print(h_i.get_shape())
                h_i = tf.concat([h_i, skip_], -1)
            else:
                h_i = tf.tanh(h_i)

        wave = h_i
        print("여기",wave.get_shape())
        print('Amount of skip connections: ', len(skips))
        print('Last wave shape: ', wave.get_shape())
        print('*************************')
        # ret feats contains the features refs to be returned
        ret_feats = [wave]
        print(ret_feats)
        ret_feats.append(z)
        print(ret_feats)
        ret_feats += alphas
        print(ret_feats)
        return ret_feats


loader = Loader()
cr, ns = loader.get_dataset()
"""
G = generator(data1)
D = discriminator(data2)
"""
with tf.Session() as sess:
    x = cr.get_next()
    z = ns.get_next()
    k = tf.convert_to_tensor(sess.run(x))

    G = generator(k)
