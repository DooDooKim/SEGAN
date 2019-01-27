import tensorflow as tf
import time
import numpy as np
from ops import *
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from make_tfrecord import Loader
from scipy.io import wavfile
import os

class SEGAN(object):
    def __init__(self, sess):
        self.batch_size = 32
        self.epoch = 150
        self.canvas_size = 2 ** 14
        self.g_learning_rate = 0.0002
        self.d_learning_rate = 0.0002
        self.l1_lambda = 100
        self.data_len = 22263
        self.beta1 = 0.5
        self.preemph = 0.95
        self.log_dir = './logs'
        self.model_dir = './model'
        self.sess = sess

    def generator(self, noisy_w, reuse=False):
        alphas = []
        skips = []
        batch_size = self.batch_size
        in_dims = noisy_w.get_shape().as_list()
        h_i = noisy_w

        if len(in_dims) == 2:
            h_i = tf.expand_dims(noisy_w, -1)
        elif len(in_dims) < 2 or len(in_dims) > 3:
            raise ValueError('Input error')

        with tf.variable_scope('generator', reuse=reuse):
            #ENCODER
            # enc ~ [16384x1, 8192x16, 4096x32, 2048x32, 1024x64, 512x64, 256x128, 128x128, 64x256, 32x256, 16x512, 8x1024]
            # dec ~ [8x2048, 16x1024, 32x512, 64x512, 8x256, 256x256, 512x128, 1024x128, 2048x64, 4096x64, 8192x32, 16384x1]
            g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
            bias_init = tf.constant_initializer(0.)
            for layer_idx, layer_depth in enumerate(g_enc_depths):
                h_i_dwn = downconv(h_i, layer_depth, kwidth=31,
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

            #concat c & z
            z = make_z([h_i.get_shape().as_list()[0], h_i.get_shape().as_list()[1], g_enc_depths[-1]])
            #z = make_z([batch_size, h_i.get_shape().as_list()[1], g_enc_depths[-1]])
            h_i = tf.concat([z, h_i], -1)

            #DECODER
            g_dec_depths = [512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 1]
            for layer_idx, layer_depth in enumerate(g_dec_depths):
                h_i_dim = h_i.get_shape().as_list()
                out_shape = [h_i_dim[0], h_i_dim[1] * 2, layer_depth]
                bias_init = tf.constant_initializer(0.)
                # deconv
                h_i_dcv = deconv(h_i, out_shape, kwidth=31, dilation=2,
                                 init=tf.truncated_normal_initializer(stddev=0.02),
                                 bias_init=bias_init, name='dec_{}'.format(layer_idx))
                h_i = h_i_dcv

                if layer_idx < len(g_dec_depths) - 1:
                    h_i = prelu(h_i, name='dec_prelu_{}'.format(layer_idx))
                    alpha_i = h_i[1]
                    h_i = h_i[0]
                    alphas.append(alpha_i)
                    skip_ = skips[-(layer_idx + 1)]
                    h_i = tf.concat([h_i, skip_], -1)

                else:
                    h_i = tf.tanh(h_i)
            if reuse:
                print('Amount of skip connections: ', len(skips))
                print('Test wave shape: ', h_i.get_shape())
                print('*****************************')

            else:
                print('Amount of skip connections: ', len(skips))
                print('Last wave shape: ', h_i.get_shape())
                print('*****************************')

            return h_i

    def discriminator(self, wave_in, reuse=False):
        d_num_fmaps = [32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 2048]
        #wave_in: waveform input
        # take the waveform as input "activation"
        in_dims = wave_in.get_shape().as_list()
        hi = wave_in
        if len(in_dims) == 2:
            hi = tf.expand_dims(wave_in, -1)
        elif len(in_dims) < 2 or len(in_dims) > 3:
            raise ValueError('Input error')

        with tf.variable_scope('discriminator', reuse=reuse):
            def disc_block(block_idx, input_, kwidth, nfmaps, bnorm, activation, pooling=2):
                with tf.variable_scope('d_block_{}'.format(block_idx),reuse=tf.AUTO_REUSE):
                    bias_init = tf.constant_initializer(0.)
                    hi_a = downconv(input_, nfmaps, kwidth=kwidth, pool=pooling,
                                    init=tf.truncated_normal_initializer(stddev=0.02),
                                    bias_init=bias_init
                                    )
                    hi_a = vbn(hi_a, 'd_vbn_{}'.format(block_idx))
                    hi =  leakyrelu(hi_a, name='lrelu')

                    return hi

            # apply input noisy layer to real and fake samples
            hi = gaussian_noise_layer(hi, 0.)

            for block_idx, fmaps in enumerate(d_num_fmaps):
                hi = disc_block(block_idx, hi, 31,
                                d_num_fmaps[block_idx],
                                True, 'leakyrelu')
            #hi_f = flatten(hi)
            #hi_f = tf.nn.dropout(hi_f, self.keep_prob_var)
            d_logit_out = conv1d(hi, kwidth=1, num_kernels=1,
                                 init=tf.truncated_normal_initializer(stddev=0.02),
                                 name='logits_conv')
            d_logit_out = tf.squeeze(d_logit_out)
            d_logit_out = fully_connected(d_logit_out, 1, activation_fn=None)

            if reuse==False:
                print('discriminator output shape: ', d_logit_out.get_shape())
                print('*****************************')
            return d_logit_out

    def build_model(self):
        load = Loader()
        cr_load_tr, ns_load_tr = load.get_dataset()

        clean_wav = cr_load_tr.get_next()
        self.clean_wav = tf.convert_to_tensor(self.sess.run(clean_wav))

        noisy_wav = ns_load_tr.get_next()
        self.noisy_wav = tf.convert_to_tensor(self.sess.run(noisy_wav))

        fm, wav_data = wavfile.read('./test/test.wav')
        wave = (2. / 65535.) * (wav_data.astype(np.float32) - 32767) + 1.
        x_pholder, preemph_op = pre_emph_test(0.95, wave.shape[0])
        wave = self.sess.run(preemph_op, feed_dict={x_pholder: wave})
        sam_wav = signal_slice(wave, 16384, 1)

        self.sam_wav = tf.convert_to_tensor(sam_wav)

        G = self.generator(self.noisy_wav)

        D_rl_concat = tf.concat([self.clean_wav, self.noisy_wav], axis=2)
        D_fk_concat = tf.concat([G, self.noisy_wav], axis=2)

        d_rl_logits = self.discriminator(D_rl_concat)
        d_fk_logits = self.discriminator(D_fk_concat, reuse=True)

        #self.d_loss = discriminator_loss(real=d_rl_logits, fake=d_fk_logits)
        self.d_loss = tf.reduce_sum(tf.square(d_rl_logits-1) + tf.square(d_fk_logits))/2

        self.g_l1_loss = self.l1_lambda * tf.reduce_mean(tf.abs(tf.subtract(G, self.clean_wav)))

        #self.g_loss = generator_loss(fake=d_fk_logits)+self.g_l1_loss
        self.g_loss = tf.reduce_sum(tf.square(d_fk_logits-1))/2

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        self.d_opt = tf.train.AdamOptimizer(self.d_learning_rate,
                                            beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
        self.g_opt = tf.train.AdamOptimizer(self.g_learning_rate,
                                            beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)

        self.test_wave = self.generator(self.sam_wav, reuse=True)

        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

    def train(self):
        counter = 1
        wav_idx = 1
        #initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        start_time = time.time()

        for epoch in range(0, self.epoch):
            # get batch data
            for idx in range(0, self.data_len-1):
                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_opt, self.d_loss_sum, self.d_loss])
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_opt, self.g_loss_sum, self.g_loss])
                self.writer.add_summary(summary_str, counter)

                counter += 1
                # display training status
                if idx % 10 == 0:
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, idx, self.data_len, time.time() - start_time, d_loss, g_loss))

                if np.mod(counter, 500) == 0:
                    self.saver.save(self.sess, './saves/model_%d.ckpt' % counter)

                    samples = self.sess.run(self.test_wave)
                    wav_r, wav_h, _ = samples.shape
                    array = np.reshape(samples, [wav_r * wav_h])

                    print('FILE test{0}_e{1}.wav is saved'.format(wav_idx, epoch))
                    wavfile.write(os.path.join('./test/test{0}_e{1}.wav'.format(wav_idx, epoch)), 16000, array)
                    wav_idx += 1








