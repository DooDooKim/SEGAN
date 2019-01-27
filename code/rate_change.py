import os
import glob

TRAIN_DIR = '.\\data\\zeroth_korean.tar\\h_data\\train\\'
TEST_DIR = '.\\data\\zeroth_korean.tar\\h_data\\test\\'
TRAIN_OUT_DIR = '.\\data\\reindeer\\train\\'
TEST_OUT_DIR = '.\\data\\reindeer\\test\\'

def wavtransform(train=False):
    if(train):
        train_dir_path = os.listdir(TRAIN_DIR)
        for idx, files in enumerate(train_dir_path):
            files = os.path.join(TRAIN_DIR, train_dir_path[idx])
            flac_files = glob.glob(files+'/*.flac')
            out_idx = train_dir_path[idx]
            for av_idx,av_files in enumerate(flac_files):
                out_dir = TRAIN_OUT_DIR+"train_wav{0}_{1}.wav".format(out_idx, av_idx)
                os.system(f"sox {av_files} {out_dir} rate 16k")
    else:
        test_dir_path = os.listdir(TEST_DIR)
        for idx, files in enumerate(test_dir_path):
            files = os.path.join(TEST_DIR, test_dir_path[idx])
            flac_files = glob.glob(files+'/*.flac')
            out_idx = test_dir_path[idx]
            for av_idx,av_files in enumerate(flac_files):
                out_dir = TEST_OUT_DIR+"test_wav{0}_{1}.wav".format(out_idx, av_idx)
                os.system(f"sox {av_files} {out_dir} rate 16k")


wavtransform()