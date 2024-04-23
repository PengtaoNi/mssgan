import os
import argparse
import librosa
import numpy as np
from scipy.io import wavfile

import torch
from torch.distributions.uniform import Uniform

import utils
from models.Unet import Unet

def eval(opt):
    utils.set_seeds(opt)
    device = utils.get_device()

    G = Unet(opt).to(device)
    G.load_state_dict(torch.load(opt.model_path))
    G.eval()
    
    noise_dist = Uniform(torch.Tensor([-1] * opt.z_dim), torch.Tensor([1] * opt.z_dim))
    noise = noise_dist.sample()

    input, sample_rate = librosa.load(opt.input_path)

    input = librosa.util.fix_length(input, size=len(input)+opt.win_length//2)

    audio = np.expand_dims(input, axis=0).T
    spectrogram, phase = utils.compute_spectrogram(audio.squeeze(1), 512, 256)
    
    inst1_pred = np.zeros(spectrogram.shape, np.float32)
    inst2_pred = np.zeros(spectrogram.shape, np.float32)

    spectrogram_len = spectrogram.shape[1]
    for i in range(0, spectrogram_len, opt.input_w):
        if i + opt.input_w > spectrogram_len:
            i = spectrogram_len - opt.input_w
        
        spectrogram_part = spectrogram[:, i:i+opt.input_w]
        spectrogram_part = torch.from_numpy(spectrogram_part).unsqueeze(0).unsqueeze(0)

        output = G([spectrogram_part.to(device), noise.to(device)]).detach().cpu().numpy()
        inst1_pred[:, i:i+opt.input_w] = output[:, 0:1]
        inst2_pred[:, i:i+opt.input_w] = output[:, 1:2]

    inst1_pred = utils.denormalise_spectrogram(inst1_pred)
    inst1_pred = utils.spectrogramToAudioFile(inst1_pred, 512, 256, phase=np.angle(phase))
    inst2_pred = utils.denormalise_spectrogram(inst2_pred)
    inst2_pred = utils.spectrogramToAudioFile(inst2_pred, 512, 256, phase=np.angle(phase))

    os.makedirs(opt.output_path, exist_ok=True)
    wavfile.write(os.path.join(opt.output_path, 'inst1.wav'), sample_rate, inst1_pred)
    wavfile.write(os.path.join(opt.output_path, 'inst2.wav'), sample_rate, inst2_pred)

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='G')
    parser.add_argument('--input_path', type=str, default='input')
    parser.add_argument('--output_path', type=str, default='output')

    parser.add_argument('--z_dim', type=int, default=50)

    parser.add_argument('--win_length', type=int, default=512)
    parser.add_argument('--hop_length', type=int, default=256)

    opt = parser.parse_args()
    print(opt)

    opt.input_h = opt.win_length // 2
    opt.input_w = opt.input_h // 2

    return opt

if __name__ == "__main__":
    opt = get_opt()

    eval(opt)