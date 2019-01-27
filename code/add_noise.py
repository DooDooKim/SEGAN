# -*- coding: utf-8 -*-
import array
import numpy as np
import random
import os
import wave

CLEAN_PATH_TR = '.\\data\\reindeer\\cr_train\\'
CLEAN_PATH_TE = '.\\data\\reindeer\\cr_test\\'
NOISE_PATH = '.\\data\\reindeer\\cr_noise\\'

OUT_NOISY_PATH_TR = '.\\data\\reindeer\\p_train\\'
OUT_NOISY_PATH_TE = '.\\data\\reindeer\\p_test\\'
OUT_NOISE_PATH_TR = '.\\data\\reindeer\\p_noise_tr\\'
OUT_NOISE_PATH_TE = '.\\data\\reindeer\\p_noise_te\\'
snr_list = [0, 5, 10, 15]

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10 ** a)
    return noise_rms

def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

if __name__ == '__main__':
    av_list = os.listdir(CLEAN_PATH_TR)
    ns_list = os.listdir(NOISE_PATH)
    filename_idx = 1
    snr_0 = 0
    snr_5 = 0
    snr_10 = 0
    snr_15 = 0
    print("noise data writing")
    for idx in range(len(av_list)):
        rn_file = random.sample(ns_list, 1)
        clean_name = av_list[idx]
        clean_file = CLEAN_PATH_TR + av_list[idx]
        noise_file = NOISE_PATH + rn_file[0]
        snr_ran = random.sample(snr_list, 1)
        snr = snr_ran[0]

        if snr == 0:
            snr_0 += 1
        elif snr == 5:
            snr_5 += 1
        elif snr == 10:
            snr_10 += 1
        elif snr == 15:
            snr_15 += 1

        clean_wav = wave.open(clean_file, "r")
        noise_wav = wave.open(noise_file, "r")

        clean_amp = cal_amp(clean_wav)
        noise_amp = cal_amp(noise_wav)

        start = random.randint(0, len(noise_amp) - len(clean_amp))
        clean_rms = cal_rms(clean_amp)
        split_noise_amp = noise_amp[start: start + len(clean_amp)]
        noise_rms = cal_rms(split_noise_amp)

        adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)

        adjusted_noise_amp = split_noise_amp * (adjusted_noise_rms / noise_rms)
        mixed_amp = (clean_amp + adjusted_noise_amp)

        if (mixed_amp.max(axis=0) > 32767):
            mixed_amp = mixed_amp * (32767 / mixed_amp.max(axis=0))
            clean_amp = clean_amp * (32767 / mixed_amp.max(axis=0))
            adjusted_noise_amp = adjusted_noise_amp * (32767 / mixed_amp.max(axis=0))
        if idx % 2000 == 0:
            print(idx,"/",len(av_list))
        noisy_wave = wave.Wave_write(OUT_NOISY_PATH_TR+clean_name)
        noisy_wave.setparams(clean_wav.getparams())
        noisy_wave.writeframes(array.array('h', mixed_amp.astype(np.int16)).tostring())
        noisy_wave.close()

        noise_wave = wave.Wave_write(OUT_NOISE_PATH_TR+"noise_tr%d.wav"%filename_idx)
        noise_wave.setparams(clean_wav.getparams())
        noise_wave.writeframes(array.array('h', adjusted_noise_amp.astype(np.int16)).tostring())
        noise_wave.close()
        filename_idx += 1
    print("done")

print("#of0:",snr_0,"#of5:",snr_5,"#of10:",snr_10,"#of15:",snr_15)
