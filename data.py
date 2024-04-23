import os
import numpy as np
from scipy.io import wavfile
import librosa

import torch
from torch.utils.data.dataset import Dataset

import utils

def preprocess(path, input_w, inst_list):
    print('Preprocessing dataset...')

    # concatenate samples for each instrument
    concat_dict = dict()
    max_len = 0
    for inst in inst_list:
        concat = []
        for wav in os.listdir(os.path.join(path, inst)):
            if not wav.endswith('.wav'):
                continue
            data, sample_rate = librosa.load(os.path.join(path, inst, wav))
            concat.append(data)
        concat = np.concatenate(concat)
        concat_dict[inst] = concat

        if len(concat) > max_len:
            max_len = len(concat)
    
    # generate mixture
    mixture = []
    for inst in inst_list:
        concat = concat_dict[inst]
        while len(concat) < max_len:
            concat = np.concatenate([concat, concat])
        concat = concat[:max_len]
        mixture.append(concat)
        concat_dict[inst] = concat
        mixture.append(concat)
    mixture = np.sum(mixture, axis=0)
    
    for inst in inst_list:
        wavfile.write(os.path.join(path, inst+'.wav'), sample_rate, concat_dict[inst])
    wavfile.write(os.path.join(path, 'mixture.wav'), sample_rate, mixture)
    
    # generate spectrograms
    inst_spectrograms = dict()
    for inst in inst_list:
        audio = np.expand_dims(concat_dict[inst], axis=0).T
        mag, phase = utils.compute_spectrogram(audio.squeeze(1), 512, 256)
        inst_spectrograms[inst] = mag
    audio = np.expand_dims(mixture, axis=0).T
    mix_spectrogram, phase = utils.compute_spectrogram(audio.squeeze(1), 512, 256)

    length = mag.shape[1]
    n_samples = 0
    for i in range(0, length-input_w, input_w//2):
        sample = []
        for inst in inst_list:
            sample.append(inst_spectrograms[inst][:, i:i+input_w])
        sample.append(mix_spectrogram[:, i:i+input_w])
        sample = np.stack(sample, axis=0)
        np.save(os.path.join(path, str(n_samples)+".npy"), sample)

        n_samples += 1

    print(f'{n_samples} samples')
    
    return n_samples

class InstDataset(Dataset):

    def __init__(self, path, input_w, inst1, inst2):
        self.path = path
        self.inst_list = [inst1, inst2]
        self.n_samples = preprocess(path, input_w, self.inst_list)

    def __getitem__(self, index):
        sample_path = os.path.join(self.path, str(index)+".npy")
        sample = torch.from_numpy(np.load(sample_path))

        return sample

    def __len__(self):
        return self.n_samples