# Acoustic-based-Cognitive-Classification

This package provides tools for acoustic feature extraction using the Python `librosa` package and the calculation of cognitive acoustic feature importance.

## Acoustic Feature Extraction

The extraction process involves 481 statistical vocal measures derived from 12 sets of acoustic variables, including:

- **Amplitude**
- **Root Mean Square (RMS)**
- **Spectrogram Polynomial Coefficients (order 0)**
- **Spectral Bandwidth**
- **Spectral Centroid**
- **Spectral Flatness**
- **Roll-off Frequency**
- **Zero-Crossing Rate**
- **Tempo**
- **Chroma Energy Normalized (CENS)**
- **Mel-Frequency Cepstral Coefficients (MFCC)**
- **MFCC Delta**

These features capture various characteristics of the speech signal, forming the basis for downstream cognitive analysis.

## Cognitive Feature Importance

To determine the importance of voice features, we employed a Random Forest Regression model with an ensemble of 100 decision tree estimators. The feature importance values were then normalized to the interval [0, 1]. The final output, containing the features and their corresponding importance scores, is saved as a CSV file.
