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
    noise = noise_dist.sample().to(device)

    input, sample_rate = librosa.load(opt.input_path)

    audio = np.expand_dims(input, axis=0).T
    spectrogram, phase = utils.compute_spectrogram(audio.squeeze(1), 512, 256)
    spectrogram = spectrogram[:, :opt.input_w]

    output = G([torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).to(device), noise])
    inst1 = output[:, 0:1]
    inst2 = output[:, 1:2]

    os.makedirs(opt.output_path, exist_ok=True)
    wavfile.write(os.path.join(opt.output_path, 'inst1.wav'), sample_rate, inst1)
    wavfile.write(os.path.join(opt.output_path, 'inst2.wav'), sample_rate, inst2)

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='G')
    parser.add_argument('--input_path', type=str, default='input')
    parser.add_argument('--output_path', type=str, default='output')

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
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