'''
This file contains functions for feature extractions
Author: Kaniel Vicencio
'''

import librosa
import numpy as np
import pandas as pd

# Waveform
def get_waveform(fpath): 
    signal, sample_rate = librosa.load(fpath)
    return signal, sample_rate

# feature extraction: mel-freq cepstral coefficients 
def get_mfccs(y, sr):
    df_mfccs = pd.DataFrame()

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    for n_mfcc in range(len(mfccs)):
        df_mfccs['MFCC_%d'%(n_mfcc+1)] = mfccs.T[n_mfcc]
    
    # visualize mfccs (for debugging)
    # print(df_mfccs.head())
    # librosa.display.specshow(mfccs, sr=sr, x_axis='time', y_axis='log')

# feature extraction: mel spectogram
import librosa.feature

def get_melspectrogram(y, sr):
    df_melspectrogram = pd.DataFrame()
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=12)
    for n_mel in range(len(mel_spectrogram)):
        df_melspectrogram['Mel_Spectogram_%d'%(n_mel+1)] = mel_spectrogram.T[n_mel]

    print(df_melspectrogram.head())
    librosa.display.specshow(mel_spectrogram, sr=sr, x_axis='time', y_axis='log')

