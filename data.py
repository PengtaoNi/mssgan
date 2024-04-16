import os
import numpy as np
from scipy.io import wavfile
import librosa

import torch
from torch.utils.data.dataset import Dataset

import utils

def preprocess(path):
    print('Preprocessing dataset...')
    inst_list = ['flute', 'oboe']

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
    interval = 10
    n_samples = max_len // interval
    for i in range(n_samples):
        for inst in inst_list:
            audio = concat_dict[inst][i*interval*sample_rate: (i+1)*interval*sample_rate]
            mag, phase = utils.compute_spectrogram(audio, 512, 256)
            np.save(os.path.join(path, inst+str(i)+'.npy'), mag)
        audio = mixture[i*interval*sample_rate: (i+1)*interval*sample_rate]
        mag, phase = utils.compute_spectrogram(audio, 512, 256)
        np.save(os.path.join(path, 'mixture'+str(i)+'.npy'), mag)
    
    return inst_list, n_samples

class InstDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.inst_list, self.n_samples = preprocess(path)

    def __getitem__(self, index):
        insts = []
        for inst in self.inst_list:
            sample_path = os.path.join(self.path, inst+str(index)+'.npy')
            insts.append(torch.from_numpy(np.load(sample_path)))
        mixture_path = os.path.join(self.path, 'mixture'+str(index)+'.npy')
        mixture = torch.from_numpy(np.load(mixture_path))

        return insts, mixture

    def __len__(self):
        return self.n_samples