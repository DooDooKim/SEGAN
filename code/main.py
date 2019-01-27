from segan_adam import *


def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sg = SEGAN(sess)
        sg.build_model()
        sg.train()

if __name__ == "__main__":
    main()
