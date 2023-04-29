'''
This file contains a bunch of utility functions for part II for 
assignment 3

Author: Kaniel Vicencio
'''

import matplotlib.pyplot as plt
import numpy as np

import librosa 
import librosa.display
def plot_waveform(y, sr, name):

    librosa.display.waveshow(y, sr=sr)
    plt.title(name)
    plt.xlabel('Time / second')
    plt.ylabel('Amplitude')

    plt.show()

def plot_frquency(y, sr, name):
    
    k = np.arange(len(y))
    T = len(y)/sr
    freq = k/T

    DATA_0 = np.fft.fft(y)
    abs_DATA_0 = abs(DATA_0)

    plt.title(name)
    plt.plot(freq, abs_DATA_0)
    plt.xlabel('Freq / Hz')
    plt.ylabel('Amplitude / dB')
    plt.xlim([0, 1000])
    plt.show()

def plot_mfcc(y, sr, name):
    



