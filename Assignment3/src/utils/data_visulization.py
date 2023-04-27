'''
This file contains a bunch of utility functions for part II for 
assignment 3

Author: Kaniel Vicencio
'''

import matplotlib.pyplot as plt
import numpy as np
import wave 

def get_waveform(fpath):
    wav_obj = wave.open(fpath, 'r')
    return wav_obj

import librosa 
import librosa.display
def plot_waveform(wf_path, name):

    y, sr = librosa.load(wf_path, duration=10)
    librosa.display.waveshow(y, sr=sr)
    plt.title(name)


