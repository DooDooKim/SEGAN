from make_tfrecord import Loader
import tensorflow as tf

loader = Loader()
cr, ns = loader.get_dataset()

with tf.Session() as sess:
    x = cr.get_next()
    sample = tf.convert_to_tensor(sess.run(x))
    print(sess.run(sample))
