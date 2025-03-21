"""
Acoustic feature extraction was performed using 481 statistical vocal measures derived from 12 sets of 
acoustic variables. These variables include:
amplitude, root mean square, spectrogram polynomial coefficients (order 0), spectral bandwidth,
spectral centroid, spectral flatness, roll-off frequency, zero-crossing rate, tempo, Chroma Energy
Normalized (CENS), Mel-Frequency Cepstral Coefficients (MFCC), and MFCC delta.
"""
import os
from datetime import datetime
import librosa
import numpy as np
import pandas as pd


def extract_features(**kwargs):
    """
    extract acoustic features
    """
    additional_cols = kwargs.get('additional_cols', [])
    voice_features = kwargs.get('voice_features')
    processing_functions = kwargs.get('processing_functions')
    input_csv = kwargs.get('input_csv', 'input.csv')
    extract_features_csv_out = kwargs.get('extract_features_csv_out', 'input_features.csv')
    load_existing = kwargs.get('load_existing', False)
    ## load existing features for a given input file
    write_feature_files = kwargs.get('write_feature_files', False)
    ## write features to CSV per input file
    speech_path_idx = kwargs.get('speech_path_idx', 'speech_path')
    feat_parent = kwargs.get('feat_parent')
    if feat_parent is not None and not os.path.isdir(feat_parent):
        os.makedirs(feat_parent)
    df = pd.read_csv(input_csv, sep=',')
    num_samples = len(df.index)  #number of speech files
    data = np.zeros((num_samples, len(voice_features)), dtype=object)
    for i in range(num_samples):
        print(datetime.now())
        print(f'Processing case {i+1} of {num_samples}')
        voice_path = df.loc[i, speech_path_idx]
        voice_fn = os.path.splitext(os.path.basename(voice_path))[0]
        feat_csv_out = os.path.join(feat_parent, f'{voice_fn}.csv')
        if load_existing and os.path.isfile(feat_csv_out):
            feat_data = pd.read_csv(feat_csv_out).to_dict(orient='records')[0]
            for j, col in enumerate(voice_features):
                data[i, j] = feat_data[col]
        else:
            x, sr = librosa.load(voice_path)
            for j, col in enumerate(voice_features):
                data[i, j] = processing_functions[col](x, sr)
            if write_feature_files:
                to_write = {col: data[i, j] for j, col in enumerate(voice_features)}
                pd.DataFrame([to_write]).to_csv(feat_csv_out, index=False)
                print(f'wrote {feat_csv_out}')
    data_df = pd.DataFrame(data, columns=voice_features)
    df.join(data_df).to_csv(extract_features_csv_out, sep=',', index=False)
