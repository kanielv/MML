'''
This file contains functions for feature extractions
Author: Kaniel Vicencio
'''

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO: Scale each feature

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

    return df_mfccs, mfccs

# feature extraction: mel spectogram

def get_melspectrogram(y, sr):
    df_melspectrogram = pd.DataFrame()
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=12)
    for n_mel in range(len(mel_spectrogram)):
        df_melspectrogram['Mel_Spectogram_%d'%(n_mel+1)] = mel_spectrogram.T[n_mel]

    return df_melspectrogram, mel_spectrogram

# feature extraction: chroma
def get_chromagram(y, sr):
    df_chroma = pd.DataFrame()
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    for n_chroma in range(len(chromagram)):
        df_chroma['Chroma_%d'%(n_chroma+1)] = chromagram.T[n_chroma]

    return df_chroma, chromagram

# generate a feature matix with the 3 features 
def generate_feature_matrix(df_mfccs, df_chroma, df_melspectrogram):
    feature_matrix = pd.concat([df_mfccs, df_chroma, df_melspectrogram], axis=1)
    return feature_matrix

