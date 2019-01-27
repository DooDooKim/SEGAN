import os
import glob

NOISE_DIR = '.\\data\\noise\\'
NOISE_OUT_DIR = '.\\data\\reindeer\\noise\\'


def noisetransform():
    noise_dir_path = os.listdir(NOISE_DIR)
    for idx, files in enumerate(noise_dir_path):
        files = os.path.join(NOISE_DIR, noise_dir_path[idx])
        flac_files = glob.glob(files+'/*.wav')
        out_idx = noise_dir_path[idx]
        for av_idx,av_files in enumerate(flac_files):
            out_dir = NOISE_OUT_DIR+"{0}_{1}.wav".format(out_idx, av_idx+1)
            os.system(f"sox {av_files} {out_dir} rate 16k")
noisetransform()
