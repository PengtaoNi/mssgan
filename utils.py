import random

import numpy as np
import torch
import librosa

def set_seeds(opt):
    '''
    Set Python, numpy as and torch random seeds to a fixed number
    :param opt: Option dictionary containined .seed member value
    '''
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

# def audio_to_spectrogram(audio, hop_length, win_length, cut_last_freq=True):
#     stft = librosa.stft(audio, hop_length=hop_length, win_length=win_length)
#     mag, phase = librosa.magphase(stft)
    
#     if cut_last_freq:
#         # Throw away last freq bin to make it number of freq bins a power of 2
#         mag = mag[:-1, :]
    
#     return np.log1p(mag), phase

def compute_spectrogram(audio, fft_size, hop_size):
    '''
    Compute magnitude spectrogram for audio signal
    :param audio: Audio input signal
    :param fft_size: FFT Window size (samples)
    :param hop_size: Hop size (samples) for STFT
    :return: Magnitude spectrogram
    '''
    stft = librosa.stft(audio, hop_length=hop_size, win_length=fft_size)
    mag, ph = librosa.magphase(stft)

    return normalise_spectrogram(mag), ph

def normalise_spectrogram(mag, cut_last_freq=True):
    '''
    Normalise audio spectrogram with log-normalisation
    :param mag: Magnitude spectrogram to be normalised
    :param cut_last_freq: Whether to cut highest frequency bin to reach power of 2 in number of bins
    :return: Normalised spectrogram
    '''
    if cut_last_freq:
        # Throw away last freq bin to make it number of freq bins a power of 2
        out = mag[:-1,:]

    # Normalize with log1p
    out = np.log1p(out)
    return out

def denormalise_spectrogram(mag, pad_freq=True):
    '''
    Reverses normalisation performed in "normalise_spectrogram" function
    :param mag: Normalised magnitudes
    :param pad_freq: Whether to append a frequency bin as highest frequency with 0 as energy
    :return: Reconstructed spectrogram
    '''
    out = np.expm1(mag)

    if pad_freq:
        out = np.pad(out, [(0,1), (0, 0)], mode="constant")

    return out

def spectrogramToAudioFile(magnitude, fftWindowSize, hopSize, phaseIterations=10, phase=None, length=None):
    '''
    Computes an audio signal from the given magnitude spectrogram, and optionally an initial phase.
    Griffin-Lim is executed to recover/refine the given the phase from the magnitude spectrogram.
    :param magnitude: Magnitudes to be converted to audio
    :param fftWindowSize: Size of FFT window used to create magnitudes
    :param hopSize: Hop size in frames used to create magnitudes
    :param phaseIterations: Number of Griffin-Lim iterations to recover phase
    :param phase: If given, starts ISTFT with this particular phase matrix
    :param length: If given, audio signal is clipped/padded to this number of frames
    :return:
    '''
    if phase is not None:
        if phaseIterations > 0:
            # Refine audio given initial phase with a number of iterations
            return reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations, phase, length)
        # reconstructing the new complex matrix
        stftMatrix = magnitude * np.exp(phase * 1j) # magnitude * e^(j*phase)
        audio = librosa.istft(stftMatrix, hop_length=hopSize, length=length)
    else:
        audio = reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations)
    return audio

def reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations=10, initPhase=None, length=None):
    '''
    Griffin-Lim algorithm for reconstructing the phase for a given magnitude spectrogram, optionally with a given
    intial phase.
    :param magnitude: Magnitudes to be converted to audio
    :param fftWindowSize: Size of FFT window used to create magnitudes
    :param hopSize: Hop size in frames used to create magnitudes
    :param phaseIterations: Number of Griffin-Lim iterations to recover phase
    :param initPhase: If given, starts reconstruction with this particular phase matrix
    :param length: If given, audio signal is clipped/padded to this number of frames
    :return:
    '''
    for i in range(phaseIterations):
        if i == 0:
            if initPhase is None:
                reconstruction = np.random.random_sample(magnitude.shape) + 1j * (2 * np.pi * np.random.random_sample(magnitude.shape) - np.pi)
            else:
                reconstruction = np.exp(initPhase * 1j) # e^(j*phase), so that angle => phase
        else:
            reconstruction = librosa.stft(audio, hop_length=hopSize, win_length=fftWindowSize)
        spectrum = magnitude * np.exp(1j * np.angle(reconstruction))
        if i == phaseIterations - 1:
            audio = librosa.istft(spectrum, hop_length=hopSize, win_length=fftWindowSize)
        else:
            audio = librosa.istft(spectrum, hop_length=hopSize, win_length=fftWindowSize)
    return audio