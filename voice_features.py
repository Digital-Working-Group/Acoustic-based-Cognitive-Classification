"""
voice_features.py
functions for calculating voice features
return data structure that maps the names of voice features
and the functions for calculating the feature
"""
import numpy as np
import librosa

def get_voice_features():
    """
    return list of voice features
    """
    voice_features = [
              'amp_min','amp_max', 'amp_range', 'amp_mean', 'amp_std', 'amp_median', 'amp_median_min', 'amp_median_max',
              'rms_min', 'rms_max', 'rms_range', 'rms_mean', 'rms_std', 'rms_median', 'rms_median_min', 'rms_median_max',
              'poly_features_min', 'poly_features_max', 'poly_features_range', 'poly_features_mean',
              'poly_features_std', 'poly_features_median', 'poly_features_median_min', 'poly_features_median_max',
              'spectral_bandwidth_min', 'spectral_bandwidth_max', 'spectral_bandwidth_range', 'spectral_bandwidth_mean', 
              'spectral_bandwidth_std', 'spectral_bandwidth_median', 'spectral_bandwidth_median_min', 'spectral_bandwidth_median_max',
              'spectral_centroid_min', 'spectral_centroid_max', 'spectral_centroid_range', 'spectral_centroid_mean', 
              'spectral_centroid_std', 'spectral_centroid_median', 'spectral_centroid_median_min', 'spectral_centroid_median_max',
              'spectral_flatness_min', 'spectral_flatness_max', 'spectral_flatness_range', 'spectral_flatness_mean', 
              'spectral_flatness_std', 'spectral_flatness_median', 'spectral_flatness_median_min', 'spectral_flatness_median_max',
              'spectral_rolloff_min', 'spectral_rolloff_max', 'spectral_rolloff_range', 'spectral_rolloff_mean', 
              'spectral_rolloff_std', 'spectral_rolloff_median', 'spectral_rolloff_median_min', 'spectral_rolloff_median_max',
              'zero_crossing_rate_min', 'zero_crossing_rate_max', 'zero_crossing_rate_range', 'zero_crossing_rate_mean', 
              'zero_crossing_rate_std', 'zero_crossing_rate_median', 'zero_crossing_rate_median_min', 'zero_crossing_rate_median_max',
              'tempo',
              'chroma_cens_0_min', 'chroma_cens_0_max', 'chroma_cens_0_range', 'chroma_cens_0_mean',
              'chroma_cens_0_std', 'chroma_cens_0_median', 'chroma_cens_0_median_min', 'chroma_cens_0_median_max',
              'chroma_cens_1_min', 'chroma_cens_1_max', 'chroma_cens_1_range', 'chroma_cens_1_mean',
              'chroma_cens_1_std', 'chroma_cens_1_median', 'chroma_cens_1_median_min', 'chroma_cens_1_median_max',
              'chroma_cens_2_min', 'chroma_cens_2_max', 'chroma_cens_2_range', 'chroma_cens_2_mean',
              'chroma_cens_2_std', 'chroma_cens_2_median', 'chroma_cens_2_median_min', 'chroma_cens_2_median_max',
              'chroma_cens_3_min', 'chroma_cens_3_max', 'chroma_cens_3_range', 'chroma_cens_3_mean',
              'chroma_cens_3_std', 'chroma_cens_3_median', 'chroma_cens_3_median_min', 'chroma_cens_3_median_max',
              'chroma_cens_4_min', 'chroma_cens_4_max', 'chroma_cens_4_range', 'chroma_cens_4_mean',
              'chroma_cens_4_std', 'chroma_cens_4_median', 'chroma_cens_4_median_min', 'chroma_cens_4_median_max',
              'chroma_cens_5_min', 'chroma_cens_5_max', 'chroma_cens_5_range', 'chroma_cens_5_mean',
              'chroma_cens_5_std', 'chroma_cens_5_median', 'chroma_cens_5_median_min', 'chroma_cens_5_median_max',
              'chroma_cens_6_min', 'chroma_cens_6_max', 'chroma_cens_6_range', 'chroma_cens_6_mean',
              'chroma_cens_6_std', 'chroma_cens_6_median', 'chroma_cens_6_median_min', 'chroma_cens_6_median_max',
              'chroma_cens_7_min', 'chroma_cens_7_max', 'chroma_cens_7_range', 'chroma_cens_7_mean',
              'chroma_cens_7_std', 'chroma_cens_7_median', 'chroma_cens_7_median_min', 'chroma_cens_7_median_max',
              'chroma_cens_8_min', 'chroma_cens_8_max', 'chroma_cens_8_range', 'chroma_cens_8_mean',
              'chroma_cens_8_std', 'chroma_cens_8_median', 'chroma_cens_8_median_min', 'chroma_cens_8_median_max',
              'chroma_cens_9_min', 'chroma_cens_9_max', 'chroma_cens_9_range', 'chroma_cens_9_mean',
              'chroma_cens_9_std', 'chroma_cens_9_median', 'chroma_cens_9_median_min', 'chroma_cens_9_median_max',
              'chroma_cens_10_min', 'chroma_cens_10_max', 'chroma_cens_10_range', 'chroma_cens_10_mean',
              'chroma_cens_10_std', 'chroma_cens_10_median', 'chroma_cens_10_median_min', 'chroma_cens_10_median_max',
              'chroma_cens_11_min', 'chroma_cens_11_max', 'chroma_cens_11_range', 'chroma_cens_11_mean',
              'chroma_cens_11_std', 'chroma_cens_11_median', 'chroma_cens_11_median_min', 'chroma_cens_11_median_max',
              'mfcc_0_min', 'mfcc_0_max', 'mfcc_0_range', 'mfcc_0_mean',
              'mfcc_0_std', 'mfcc_0_median', 'mfcc_0_median_min', 'mfcc_0_median_max',
              'mfcc_1_min', 'mfcc_1_max', 'mfcc_1_range', 'mfcc_1_mean',
              'mfcc_1_std', 'mfcc_1_median', 'mfcc_1_median_min', 'mfcc_1_median_max',
              'mfcc_2_min', 'mfcc_2_max', 'mfcc_2_range', 'mfcc_2_mean',
              'mfcc_2_std', 'mfcc_2_median', 'mfcc_2_median_min', 'mfcc_2_median_max',
              'mfcc_3_min', 'mfcc_3_max', 'mfcc_3_range', 'mfcc_3_mean',
              'mfcc_3_std', 'mfcc_3_median', 'mfcc_3_median_min', 'mfcc_3_median_max',
              'mfcc_4_min', 'mfcc_4_max', 'mfcc_4_range', 'mfcc_4_mean',
              'mfcc_4_std', 'mfcc_4_median', 'mfcc_4_median_min', 'mfcc_4_median_max',
              'mfcc_5_min', 'mfcc_5_max', 'mfcc_5_range', 'mfcc_5_mean',
              'mfcc_5_std', 'mfcc_5_median', 'mfcc_5_median_min', 'mfcc_5_median_max',
              'mfcc_6_min', 'mfcc_6_max', 'mfcc_6_range', 'mfcc_6_mean',
              'mfcc_6_std', 'mfcc_6_median', 'mfcc_6_median_min', 'mfcc_6_median_max',
              'mfcc_7_min', 'mfcc_7_max', 'mfcc_7_range', 'mfcc_7_mean',
              'mfcc_7_std', 'mfcc_7_median', 'mfcc_7_median_min', 'mfcc_7_median_max',
              'mfcc_8_min', 'mfcc_8_max', 'mfcc_8_range', 'mfcc_8_mean',
              'mfcc_8_std', 'mfcc_8_median', 'mfcc_8_median_min', 'mfcc_8_median_max',
              'mfcc_9_min', 'mfcc_9_max', 'mfcc_9_range', 'mfcc_9_mean',
              'mfcc_9_std', 'mfcc_9_median', 'mfcc_9_median_min', 'mfcc_9_median_max',
              'mfcc_10_min', 'mfcc_10_max', 'mfcc_10_range', 'mfcc_10_mean',
              'mfcc_10_std', 'mfcc_10_median', 'mfcc_10_median_min', 'mfcc_10_median_max',
              'mfcc_11_min', 'mfcc_11_max', 'mfcc_11_range', 'mfcc_11_mean',
              'mfcc_11_std', 'mfcc_11_median', 'mfcc_11_median_min', 'mfcc_11_median_max',
              'mfcc_12_min', 'mfcc_12_max', 'mfcc_12_range', 'mfcc_12_mean',
              'mfcc_12_std', 'mfcc_12_median', 'mfcc_12_median_min', 'mfcc_12_median_max',
              'mfcc_13_min', 'mfcc_13_max', 'mfcc_13_range', 'mfcc_13_mean',
              'mfcc_13_std', 'mfcc_13_median', 'mfcc_13_median_min', 'mfcc_13_median_max',
              'mfcc_14_min', 'mfcc_14_max', 'mfcc_14_range', 'mfcc_14_mean',
              'mfcc_14_std', 'mfcc_14_median', 'mfcc_14_median_min', 'mfcc_14_median_max',
              'mfcc_15_min', 'mfcc_15_max', 'mfcc_15_range', 'mfcc_15_mean',
              'mfcc_15_std', 'mfcc_15_median', 'mfcc_15_median_min', 'mfcc_15_median_max',
              'mfcc_16_min', 'mfcc_16_max', 'mfcc_16_range', 'mfcc_16_mean',
              'mfcc_16_std', 'mfcc_16_median', 'mfcc_16_median_min', 'mfcc_16_median_max',
              'mfcc_17_min', 'mfcc_17_max', 'mfcc_17_range', 'mfcc_17_mean',
              'mfcc_17_std', 'mfcc_17_median', 'mfcc_17_median_min', 'mfcc_17_median_max',
              'mfcc_18_min', 'mfcc_18_max', 'mfcc_18_range', 'mfcc_18_mean',
              'mfcc_18_std', 'mfcc_18_median', 'mfcc_18_median_min', 'mfcc_18_median_max',
              'mfcc_19_min', 'mfcc_19_max', 'mfcc_19_range', 'mfcc_19_mean',
              'mfcc_19_std', 'mfcc_19_median', 'mfcc_19_median_min', 'mfcc_19_median_max',
              'delta_0_min', 'delta_0_max', 'delta_0_range', 'delta_0_mean',
              'delta_0_std', 'delta_0_median', 'delta_0_median_min', 'delta_0_median_max',
              'delta_1_min', 'delta_1_max', 'delta_1_range', 'delta_1_mean',
              'delta_1_std', 'delta_1_median', 'delta_1_median_min', 'delta_1_median_max',
              'delta_2_min', 'delta_2_max', 'delta_2_range', 'delta_2_mean',
              'delta_2_std', 'delta_2_median', 'delta_2_median_min', 'delta_2_median_max',
              'delta_3_min', 'delta_3_max', 'delta_3_range', 'delta_3_mean',
              'delta_3_std', 'delta_3_median', 'delta_3_median_min', 'delta_3_median_max',
              'delta_4_min', 'delta_4_max', 'delta_4_range', 'delta_4_mean',
              'delta_4_std', 'delta_4_median', 'delta_4_median_min', 'delta_4_median_max',
              'delta_5_min', 'delta_5_max', 'delta_5_range', 'delta_5_mean',
              'delta_5_std', 'delta_5_median', 'delta_5_median_min', 'delta_5_median_max',
              'delta_6_min', 'delta_6_max', 'delta_6_range', 'delta_6_mean',
              'delta_6_std', 'delta_6_median', 'delta_6_median_min', 'delta_6_median_max',
              'delta_7_min', 'delta_7_max', 'delta_7_range', 'delta_7_mean',
              'delta_7_std', 'delta_7_median', 'delta_7_median_min', 'delta_7_median_max',
              'delta_8_min', 'delta_8_max', 'delta_8_range', 'delta_8_mean',
              'delta_8_std', 'delta_8_median', 'delta_8_median_min', 'delta_8_median_max',
              'delta_9_min', 'delta_9_max', 'delta_9_range', 'delta_9_mean',
              'delta_9_std', 'delta_9_median', 'delta_9_median_min', 'delta_9_median_max',
              'delta_10_min', 'delta_10_max', 'delta_10_range', 'delta_10_mean',
              'delta_10_std', 'delta_10_median', 'delta_10_median_min', 'delta_10_median_max',
              'delta_11_min', 'delta_11_max', 'delta_11_range', 'delta_11_mean',
              'delta_11_std', 'delta_11_median', 'delta_11_median_min', 'delta_11_median_max',
              'delta_12_min', 'delta_12_max', 'delta_12_range', 'delta_12_mean',
              'delta_12_std', 'delta_12_median', 'delta_12_median_min', 'delta_12_median_max',
              'delta_13_min', 'delta_13_max', 'delta_13_range', 'delta_13_mean',
              'delta_13_std', 'delta_13_median', 'delta_13_median_min', 'delta_13_median_max',
              'delta_14_min', 'delta_14_max', 'delta_14_range', 'delta_14_mean',
              'delta_14_std', 'delta_14_median', 'delta_14_median_min', 'delta_14_median_max',
              'delta_15_min', 'delta_15_max', 'delta_15_range', 'delta_15_mean',
              'delta_15_std', 'delta_15_median', 'delta_15_median_min', 'delta_15_median_max',
              'delta_16_min', 'delta_16_max', 'delta_16_range', 'delta_16_mean',
              'delta_16_std', 'delta_16_median', 'delta_16_median_min', 'delta_16_median_max',
              'delta_17_min', 'delta_17_max', 'delta_17_range', 'delta_17_mean',
              'delta_17_std', 'delta_17_median', 'delta_17_median_min', 'delta_17_median_max',
              'delta_18_min', 'delta_18_max', 'delta_18_range', 'delta_18_mean',
              'delta_18_std', 'delta_18_median', 'delta_18_median_min', 'delta_18_median_max',
              'delta_19_min', 'delta_19_max', 'delta_19_range', 'delta_19_mean',
              'delta_19_std', 'delta_19_median', 'delta_19_median_min', 'delta_19_median_max'
              ]
    return voice_features

def get_processing_functions():
    """
    mapping of voice feats to funct
    """
    processing_functions = {
        #amplitude
        'amp_min': process_amp_min,
        'amp_max': process_amp_max,
        'amp_range': process_amp_range,
        'amp_mean': process_amp_mean,
        'amp_std': process_amp_std,
        'amp_median': process_amp_median,
        'amp_median_min': process_amp_median_min,
        'amp_median_max': process_amp_median_max,

        #root-mean-square value for each frame
        'rms_min': process_rms_min,
        'rms_max': process_rms_max,
        'rms_range': process_rms_range,
        'rms_mean': process_rms_mean,
        'rms_std': process_rms_std,
        'rms_median': process_rms_median,
        'rms_median_min': process_rms_median_min,
        'rms_median_max': process_rms_median_max,

        #polynomial coefficients for each frame (order 0)
        'poly_features_min': process_poly_features_min,
        'poly_features_max': process_poly_features_max,
        'poly_features_range': process_poly_features_range,
        'poly_features_mean': process_poly_features_mean,
        'poly_features_std': process_poly_features_std,
        'poly_features_median': process_poly_features_median,
        'poly_features_median_min': process_poly_features_median_min,
        'poly_features_median_max': process_poly_features_median_max,

        #spectral bandwidth
        'spectral_bandwidth_min': process_spectral_bandwidth_min,
        'spectral_bandwidth_max': process_spectral_bandwidth_max,
        'spectral_bandwidth_range': process_spectral_bandwidth_range,
        'spectral_bandwidth_mean': process_spectral_bandwidth_mean,
        'spectral_bandwidth_std': process_spectral_bandwidth_std,
        'spectral_bandwidth_median': process_spectral_bandwidth_median,
        'spectral_bandwidth_median_min': process_spectral_bandwidth_median_min,
        'spectral_bandwidth_median_max': process_spectral_bandwidth_median_max,

        #spectral centroid
        'spectral_centroid_min': process_spectral_centroid_min,
        'spectral_centroid_max': process_spectral_centroid_max,
        'spectral_centroid_range': process_spectral_centroid_range,
        'spectral_centroid_mean': process_spectral_centroid_mean,
        'spectral_centroid_std': process_spectral_centroid_std,
        'spectral_centroid_median': process_spectral_centroid_median,
        'spectral_centroid_median_min': process_spectral_centroid_median_min,
        'spectral_centroid_median_max': process_spectral_centroid_median_max,

        #spectral flatness
        'spectral_flatness_min': process_spectral_flatness_min,
        'spectral_flatness_max': process_spectral_flatness_max,
        'spectral_flatness_range': process_spectral_flatness_range,
        'spectral_flatness_mean': process_spectral_flatness_mean,
        'spectral_flatness_std': process_spectral_flatness_std,
        'spectral_flatness_median': process_spectral_flatness_median,
        'spectral_flatness_median_min': process_spectral_flatness_median_min,
        'spectral_flatness_median_max': process_spectral_flatness_median_max,

        #spectral rolloff
        'spectral_rolloff_min': process_spectral_rolloff_min,
        'spectral_rolloff_max': process_spectral_rolloff_max,
        'spectral_rolloff_range': process_spectral_rolloff_range,
        'spectral_rolloff_mean': process_spectral_rolloff_mean,
        'spectral_rolloff_std': process_spectral_rolloff_std,
        'spectral_rolloff_median': process_spectral_rolloff_median,
        'spectral_rolloff_median_min': process_spectral_rolloff_median_min,
        'spectral_rolloff_median_max': process_spectral_rolloff_median_max,

        #zero crossing rate
        'zero_crossing_rate_min': process_zero_crossing_rate_min,
        'zero_crossing_rate_max': process_zero_crossing_rate_max,
        'zero_crossing_rate_range': process_zero_crossing_rate_range,
        'zero_crossing_rate_mean': process_zero_crossing_rate_mean,
        'zero_crossing_rate_std': process_zero_crossing_rate_std,
        'zero_crossing_rate_median': process_zero_crossing_rate_median,
        'zero_crossing_rate_median_min': process_zero_crossing_rate_median_min,
        'zero_crossing_rate_median_max': process_zero_crossing_rate_median_max,

        #tempo
        'tempo': process_tempo,

        #chromagram (pitch) bin 0
        'chroma_cens_0_min': process_chroma_cens_0_min,
        'chroma_cens_0_max': process_chroma_cens_0_max,
        'chroma_cens_0_range': process_chroma_cens_0_range,
        'chroma_cens_0_mean': process_chroma_cens_0_mean,
        'chroma_cens_0_std': process_chroma_cens_0_std,
        'chroma_cens_0_median': process_chroma_cens_0_median,
        'chroma_cens_0_median_min': process_chroma_cens_0_median_min,
        'chroma_cens_0_median_max': process_chroma_cens_0_median_max,

        #chromagram (pitch) bin 1
        'chroma_cens_1_min': process_chroma_cens_1_min,
        'chroma_cens_1_max': process_chroma_cens_1_max,
        'chroma_cens_1_range': process_chroma_cens_1_range,
        'chroma_cens_1_mean': process_chroma_cens_1_mean,
        'chroma_cens_1_std': process_chroma_cens_1_std,
        'chroma_cens_1_median': process_chroma_cens_1_median,
        'chroma_cens_1_median_min': process_chroma_cens_1_median_min,
        'chroma_cens_1_median_max': process_chroma_cens_1_median_max,

        #chromagram (pitch) bin 2
        'chroma_cens_2_min': process_chroma_cens_2_min,
        'chroma_cens_2_max': process_chroma_cens_2_max,
        'chroma_cens_2_range': process_chroma_cens_2_range,
        'chroma_cens_2_mean': process_chroma_cens_2_mean,
        'chroma_cens_2_std': process_chroma_cens_2_std,
        'chroma_cens_2_median': process_chroma_cens_2_median,
        'chroma_cens_2_median_min': process_chroma_cens_2_median_min,
        'chroma_cens_2_median_max': process_chroma_cens_2_median_max,

        #chromagram (pitch) bin 3
        'chroma_cens_3_min': process_chroma_cens_3_min,
        'chroma_cens_3_max': process_chroma_cens_3_max,
        'chroma_cens_3_range': process_chroma_cens_3_range,
        'chroma_cens_3_mean': process_chroma_cens_3_mean,
        'chroma_cens_3_std': process_chroma_cens_3_std,
        'chroma_cens_3_median': process_chroma_cens_3_median,
        'chroma_cens_3_median_min': process_chroma_cens_3_median_min,
        'chroma_cens_3_median_max': process_chroma_cens_3_median_max,

        #chromagram (pitch) bin 4
        'chroma_cens_4_min': process_chroma_cens_4_min,
        'chroma_cens_4_max': process_chroma_cens_4_max,
        'chroma_cens_4_range': process_chroma_cens_4_range,
        'chroma_cens_4_mean': process_chroma_cens_4_mean,
        'chroma_cens_4_std': process_chroma_cens_4_std,
        'chroma_cens_4_median': process_chroma_cens_4_median,
        'chroma_cens_4_median_min': process_chroma_cens_4_median_min,
        'chroma_cens_4_median_max': process_chroma_cens_4_median_max,

        #chromagram (pitch) bin 5
        'chroma_cens_5_min': process_chroma_cens_5_min,
        'chroma_cens_5_max': process_chroma_cens_5_max,
        'chroma_cens_5_range': process_chroma_cens_5_range,
        'chroma_cens_5_mean': process_chroma_cens_5_mean,
        'chroma_cens_5_std': process_chroma_cens_5_std,
        'chroma_cens_5_median': process_chroma_cens_5_median,
        'chroma_cens_5_median_min': process_chroma_cens_5_median_min,
        'chroma_cens_5_median_max': process_chroma_cens_5_median_max,

        #chromagram (pitch) bin 6
        'chroma_cens_6_min': process_chroma_cens_6_min,
        'chroma_cens_6_max': process_chroma_cens_6_max,
        'chroma_cens_6_range': process_chroma_cens_6_range,
        'chroma_cens_6_mean': process_chroma_cens_6_mean,
        'chroma_cens_6_std': process_chroma_cens_6_std,
        'chroma_cens_6_median': process_chroma_cens_6_median,
        'chroma_cens_6_median_min': process_chroma_cens_6_median_min,
        'chroma_cens_6_median_max': process_chroma_cens_6_median_max,

        #chromagram (pitch) bin 7
        'chroma_cens_7_min': process_chroma_cens_7_min,
        'chroma_cens_7_max': process_chroma_cens_7_max,
        'chroma_cens_7_range': process_chroma_cens_7_range,
        'chroma_cens_7_mean': process_chroma_cens_7_mean,
        'chroma_cens_7_std': process_chroma_cens_7_std,
        'chroma_cens_7_median': process_chroma_cens_7_median,
        'chroma_cens_7_median_min': process_chroma_cens_7_median_min,
        'chroma_cens_7_median_max': process_chroma_cens_7_median_max,

        #chromagram (pitch) bin 8
        'chroma_cens_8_min': process_chroma_cens_8_min,
        'chroma_cens_8_max': process_chroma_cens_8_max,
        'chroma_cens_8_range': process_chroma_cens_8_range,
        'chroma_cens_8_mean': process_chroma_cens_8_mean,
        'chroma_cens_8_std': process_chroma_cens_8_std,
        'chroma_cens_8_median': process_chroma_cens_8_median,
        'chroma_cens_8_median_min': process_chroma_cens_8_median_min,
        'chroma_cens_8_median_max': process_chroma_cens_8_median_max,

        #chromagram (pitch) bin 9
        'chroma_cens_9_min': process_chroma_cens_9_min,
        'chroma_cens_9_max': process_chroma_cens_9_max,
        'chroma_cens_9_range': process_chroma_cens_9_range,
        'chroma_cens_9_mean': process_chroma_cens_9_mean,
        'chroma_cens_9_std': process_chroma_cens_9_std,
        'chroma_cens_9_median': process_chroma_cens_9_median,
        'chroma_cens_9_median_min': process_chroma_cens_9_median_min,
        'chroma_cens_9_median_max': process_chroma_cens_9_median_max,

        #chromagram (pitch) bin 10
        'chroma_cens_10_min': process_chroma_cens_10_min,
        'chroma_cens_10_max': process_chroma_cens_10_max,
        'chroma_cens_10_range': process_chroma_cens_10_range,
        'chroma_cens_10_mean': process_chroma_cens_10_mean,
        'chroma_cens_10_std': process_chroma_cens_10_std,
        'chroma_cens_10_median': process_chroma_cens_10_median,
        'chroma_cens_10_median_min': process_chroma_cens_10_median_min,
        'chroma_cens_10_median_max': process_chroma_cens_10_median_max,

        #chromagram (pitch) bin 11
        'chroma_cens_11_min': process_chroma_cens_11_min,
        'chroma_cens_11_max': process_chroma_cens_11_max,
        'chroma_cens_11_range': process_chroma_cens_11_range,
        'chroma_cens_11_mean': process_chroma_cens_11_mean,
        'chroma_cens_11_std': process_chroma_cens_11_std,
        'chroma_cens_11_median': process_chroma_cens_11_median,
        'chroma_cens_11_median_min': process_chroma_cens_11_median_min,
        'chroma_cens_11_median_max': process_chroma_cens_11_median_max,

        #MFCC 0
        'mfcc_0_min': process_mfcc_0_min,
        'mfcc_0_max': process_mfcc_0_max,
        'mfcc_0_range': process_mfcc_0_range,
        'mfcc_0_mean': process_mfcc_0_mean,
        'mfcc_0_std': process_mfcc_0_std,
        'mfcc_0_median': process_mfcc_0_median,
        'mfcc_0_median_min': process_mfcc_0_median_min,
        'mfcc_0_median_max': process_mfcc_0_median_max,

        #MFCC 1
        'mfcc_1_min': process_mfcc_1_min,
        'mfcc_1_max': process_mfcc_1_max,
        'mfcc_1_range': process_mfcc_1_range,
        'mfcc_1_mean': process_mfcc_1_mean,
        'mfcc_1_std': process_mfcc_1_std,
        'mfcc_1_median': process_mfcc_1_median,
        'mfcc_1_median_min': process_mfcc_1_median_min,
        'mfcc_1_median_max': process_mfcc_1_median_max,

        #MFCC 2
        'mfcc_2_min': process_mfcc_2_min,
        'mfcc_2_max': process_mfcc_2_max,
        'mfcc_2_range': process_mfcc_2_range,
        'mfcc_2_mean': process_mfcc_2_mean,
        'mfcc_2_std': process_mfcc_2_std,
        'mfcc_2_median': process_mfcc_2_median,
        'mfcc_2_median_min': process_mfcc_2_median_min,
        'mfcc_2_median_max': process_mfcc_2_median_max,

        #MFCC 3
        'mfcc_3_min': process_mfcc_3_min,
        'mfcc_3_max': process_mfcc_3_max,
        'mfcc_3_range': process_mfcc_3_range,
        'mfcc_3_mean': process_mfcc_3_mean,
        'mfcc_3_std': process_mfcc_3_std,
        'mfcc_3_median': process_mfcc_3_median,
        'mfcc_3_median_min': process_mfcc_3_median_min,
        'mfcc_3_median_max': process_mfcc_3_median_max,

        #MFCC 4
        'mfcc_4_min': process_mfcc_4_min,
        'mfcc_4_max': process_mfcc_4_max,
        'mfcc_4_range': process_mfcc_4_range,
        'mfcc_4_mean': process_mfcc_4_mean,
        'mfcc_4_std': process_mfcc_4_std,
        'mfcc_4_median': process_mfcc_4_median,
        'mfcc_4_median_min': process_mfcc_4_median_min,
        'mfcc_4_median_max': process_mfcc_4_median_max,

        #MFCC 5
        'mfcc_5_min': process_mfcc_5_min,
        'mfcc_5_max': process_mfcc_5_max,
        'mfcc_5_range': process_mfcc_5_range,
        'mfcc_5_mean': process_mfcc_5_mean,
        'mfcc_5_std': process_mfcc_5_std,
        'mfcc_5_median': process_mfcc_5_median,
        'mfcc_5_median_min': process_mfcc_5_median_min,
        'mfcc_5_median_max': process_mfcc_5_median_max,

        #MFCC 6
        'mfcc_6_min': process_mfcc_6_min,
        'mfcc_6_max': process_mfcc_6_max,
        'mfcc_6_range': process_mfcc_6_range,
        'mfcc_6_mean': process_mfcc_6_mean,
        'mfcc_6_std': process_mfcc_6_std,
        'mfcc_6_median': process_mfcc_6_median,
        'mfcc_6_median_min': process_mfcc_6_median_min,
        'mfcc_6_median_max': process_mfcc_6_median_max,

        #MFCC 7
        'mfcc_7_min': process_mfcc_7_min,
        'mfcc_7_max': process_mfcc_7_max,
        'mfcc_7_range': process_mfcc_7_range,
        'mfcc_7_mean': process_mfcc_7_mean,
        'mfcc_7_std': process_mfcc_7_std,
        'mfcc_7_median': process_mfcc_7_median,
        'mfcc_7_median_min': process_mfcc_7_median_min,
        'mfcc_7_median_max': process_mfcc_7_median_max,

        #MFCC 8
        'mfcc_8_min': process_mfcc_8_min,
        'mfcc_8_max': process_mfcc_8_max,
        'mfcc_8_range': process_mfcc_8_range,
        'mfcc_8_mean': process_mfcc_8_mean,
        'mfcc_8_std': process_mfcc_8_std,
        'mfcc_8_median': process_mfcc_8_median,
        'mfcc_8_median_min': process_mfcc_8_median_min,
        'mfcc_8_median_max': process_mfcc_8_median_max,

        #MFCC 9
        'mfcc_9_min': process_mfcc_9_min,
        'mfcc_9_max': process_mfcc_9_max,
        'mfcc_9_range': process_mfcc_9_range,
        'mfcc_9_mean': process_mfcc_9_mean,
        'mfcc_9_std': process_mfcc_9_std,
        'mfcc_9_median': process_mfcc_9_median,
        'mfcc_9_median_min': process_mfcc_9_median_min,
        'mfcc_9_median_max': process_mfcc_9_median_max,

        #MFCC 10
        'mfcc_10_min': process_mfcc_10_min,
        'mfcc_10_max': process_mfcc_10_max,
        'mfcc_10_range': process_mfcc_10_range,
        'mfcc_10_mean': process_mfcc_10_mean,
        'mfcc_10_std': process_mfcc_10_std,
        'mfcc_10_median': process_mfcc_10_median,
        'mfcc_10_median_min': process_mfcc_10_median_min,
        'mfcc_10_median_max': process_mfcc_10_median_max,

        #MFCC 11
        'mfcc_11_min': process_mfcc_11_min,
        'mfcc_11_max': process_mfcc_11_max,
        'mfcc_11_range': process_mfcc_11_range,
        'mfcc_11_mean': process_mfcc_11_mean,
        'mfcc_11_std': process_mfcc_11_std,
        'mfcc_11_median': process_mfcc_11_median,
        'mfcc_11_median_min': process_mfcc_11_median_min,
        'mfcc_11_median_max': process_mfcc_11_median_max,

        #MFCC 12
        'mfcc_12_min': process_mfcc_12_min,
        'mfcc_12_max': process_mfcc_12_max,
        'mfcc_12_range': process_mfcc_12_range,
        'mfcc_12_mean': process_mfcc_12_mean,
        'mfcc_12_std': process_mfcc_12_std,
        'mfcc_12_median': process_mfcc_12_median,
        'mfcc_12_median_min': process_mfcc_12_median_min,
        'mfcc_12_median_max': process_mfcc_12_median_max,

        #MFCC 13
        'mfcc_13_min': process_mfcc_13_min,
        'mfcc_13_max': process_mfcc_13_max,
        'mfcc_13_range': process_mfcc_13_range,
        'mfcc_13_mean': process_mfcc_13_mean,
        'mfcc_13_std': process_mfcc_13_std,
        'mfcc_13_median': process_mfcc_13_median,
        'mfcc_13_median_min': process_mfcc_13_median_min,
        'mfcc_13_median_max': process_mfcc_13_median_max,

        #MFCC 14
        'mfcc_14_min': process_mfcc_14_min,
        'mfcc_14_max': process_mfcc_14_max,
        'mfcc_14_range': process_mfcc_14_range,
        'mfcc_14_mean': process_mfcc_14_mean,
        'mfcc_14_std': process_mfcc_14_std,
        'mfcc_14_median': process_mfcc_14_median,
        'mfcc_14_median_min': process_mfcc_14_median_min,
        'mfcc_14_median_max': process_mfcc_14_median_max,

        #MFCC 15
        'mfcc_15_min': process_mfcc_15_min,
        'mfcc_15_max': process_mfcc_15_max,
        'mfcc_15_range': process_mfcc_15_range,
        'mfcc_15_mean': process_mfcc_15_mean,
        'mfcc_15_std': process_mfcc_15_std,
        'mfcc_15_median': process_mfcc_15_median,
        'mfcc_15_median_min': process_mfcc_15_median_min,
        'mfcc_15_median_max': process_mfcc_15_median_max,

        #MFCC 16
        'mfcc_16_min': process_mfcc_16_min,
        'mfcc_16_max': process_mfcc_16_max,
        'mfcc_16_range': process_mfcc_16_range,
        'mfcc_16_mean': process_mfcc_16_mean,
        'mfcc_16_std': process_mfcc_16_std,
        'mfcc_16_median': process_mfcc_16_median,
        'mfcc_16_median_min': process_mfcc_16_median_min,
        'mfcc_16_median_max': process_mfcc_16_median_max,

        #MFCC 17
        'mfcc_17_min': process_mfcc_17_min,
        'mfcc_17_max': process_mfcc_17_max,
        'mfcc_17_range': process_mfcc_17_range,
        'mfcc_17_mean': process_mfcc_17_mean,
        'mfcc_17_std': process_mfcc_17_std,
        'mfcc_17_median': process_mfcc_17_median,
        'mfcc_17_median_min': process_mfcc_17_median_min,
        'mfcc_17_median_max': process_mfcc_17_median_max,

        #MFCC 18
        'mfcc_18_min': process_mfcc_18_min,
        'mfcc_18_max': process_mfcc_18_max,
        'mfcc_18_range': process_mfcc_18_range,
        'mfcc_18_mean': process_mfcc_18_mean,
        'mfcc_18_std': process_mfcc_18_std,
        'mfcc_18_median': process_mfcc_18_median,
        'mfcc_18_median_min': process_mfcc_18_median_min,
        'mfcc_18_median_max': process_mfcc_18_median_max,

        #MFCC 19
        'mfcc_19_min': process_mfcc_19_min,
        'mfcc_19_max': process_mfcc_19_max,
        'mfcc_19_range': process_mfcc_19_range,
        'mfcc_19_mean': process_mfcc_19_mean,
        'mfcc_19_std': process_mfcc_19_std,
        'mfcc_19_median': process_mfcc_19_median,
        'mfcc_19_median_min': process_mfcc_19_median_min,
        'mfcc_19_median_max': process_mfcc_19_median_max,

        #Delta 0
        'delta_0_min': process_delta_0_min,
        'delta_0_max': process_delta_0_max,
        'delta_0_range': process_delta_0_range,
        'delta_0_mean': process_delta_0_mean,
        'delta_0_std': process_delta_0_std,
        'delta_0_median': process_delta_0_median,
        'delta_0_median_min': process_delta_0_median_min,
        'delta_0_median_max': process_delta_0_median_max,

         #Delta 1
        'delta_1_min': process_delta_1_min,
        'delta_1_max': process_delta_1_max,
        'delta_1_range': process_delta_1_range,
        'delta_1_mean': process_delta_1_mean,
        'delta_1_std': process_delta_1_std,
        'delta_1_median': process_delta_1_median,
        'delta_1_median_min': process_delta_1_median_min,
        'delta_1_median_max': process_delta_1_median_max,

        #Delta 2
        'delta_2_min': process_delta_2_min,
        'delta_2_max': process_delta_2_max,
        'delta_2_range': process_delta_2_range,
        'delta_2_mean': process_delta_2_mean,
        'delta_2_std': process_delta_2_std,
        'delta_2_median': process_delta_2_median,
        'delta_2_median_min': process_delta_2_median_min,
        'delta_2_median_max': process_delta_2_median_max,

        #Delta 3
        'delta_3_min': process_delta_3_min,
        'delta_3_max': process_delta_3_max,
        'delta_3_range': process_delta_3_range,
        'delta_3_mean': process_delta_3_mean,
        'delta_3_std': process_delta_3_std,
        'delta_3_median': process_delta_3_median,
        'delta_3_median_min': process_delta_3_median_min,
        'delta_3_median_max': process_delta_3_median_max,

        #Delta 4
        'delta_4_min': process_delta_4_min,
        'delta_4_max': process_delta_4_max,
        'delta_4_range': process_delta_4_range,
        'delta_4_mean': process_delta_4_mean,
        'delta_4_std': process_delta_4_std,
        'delta_4_median': process_delta_4_median,
        'delta_4_median_min': process_delta_4_median_min,
        'delta_4_median_max': process_delta_4_median_max,

        #Delta 5
        'delta_5_min': process_delta_5_min,
        'delta_5_max': process_delta_5_max,
        'delta_5_range': process_delta_5_range,
        'delta_5_mean': process_delta_5_mean,
        'delta_5_std': process_delta_5_std,
        'delta_5_median': process_delta_5_median,
        'delta_5_median_min': process_delta_5_median_min,
        'delta_5_median_max': process_delta_5_median_max,

        #Delta 6
        'delta_6_min': process_delta_6_min,
        'delta_6_max': process_delta_6_max,
        'delta_6_range': process_delta_6_range,
        'delta_6_mean': process_delta_6_mean,
        'delta_6_std': process_delta_6_std,
        'delta_6_median': process_delta_6_median,
        'delta_6_median_min': process_delta_6_median_min,
        'delta_6_median_max': process_delta_6_median_max,

        #Delta 7
        'delta_7_min': process_delta_7_min,
        'delta_7_max': process_delta_7_max,
        'delta_7_range': process_delta_7_range,
        'delta_7_mean': process_delta_7_mean,
        'delta_7_std': process_delta_7_std,
        'delta_7_median': process_delta_7_median,
        'delta_7_median_min': process_delta_7_median_min,
        'delta_7_median_max': process_delta_7_median_max,

        #Delta 8
        'delta_8_min': process_delta_8_min,
        'delta_8_max': process_delta_8_max,
        'delta_8_range': process_delta_8_range,
        'delta_8_mean': process_delta_8_mean,
        'delta_8_std': process_delta_8_std,
        'delta_8_median': process_delta_8_median,
        'delta_8_median_min': process_delta_8_median_min,
        'delta_8_median_max': process_delta_8_median_max,

        #Delta 9
        'delta_9_min': process_delta_9_min,
        'delta_9_max': process_delta_9_max,
        'delta_9_range': process_delta_9_range,
        'delta_9_mean': process_delta_9_mean,
        'delta_9_std': process_delta_9_std,
        'delta_9_median': process_delta_9_median,
        'delta_9_median_min': process_delta_9_median_min,
        'delta_9_median_max': process_delta_9_median_max,

        #Delta 10
        'delta_10_min': process_delta_10_min,
        'delta_10_max': process_delta_10_max,
        'delta_10_range': process_delta_10_range,
        'delta_10_mean': process_delta_10_mean,
        'delta_10_std': process_delta_10_std,
        'delta_10_median': process_delta_10_median,
        'delta_10_median_min': process_delta_10_median_min,
        'delta_10_median_max': process_delta_10_median_max,

        #Delta 11
        'delta_11_min': process_delta_11_min,
        'delta_11_max': process_delta_11_max,
        'delta_11_range': process_delta_11_range,
        'delta_11_mean': process_delta_11_mean,
        'delta_11_std': process_delta_11_std,
        'delta_11_median': process_delta_11_median,
        'delta_11_median_min': process_delta_11_median_min,
        'delta_11_median_max': process_delta_11_median_max,

        #Delta 12
        'delta_12_min': process_delta_12_min,
        'delta_12_max': process_delta_12_max,
        'delta_12_range': process_delta_12_range,
        'delta_12_mean': process_delta_12_mean,
        'delta_12_std': process_delta_12_std,
        'delta_12_median': process_delta_12_median,
        'delta_12_median_min': process_delta_12_median_min,
        'delta_12_median_max': process_delta_12_median_max,

        #Delta 13
        'delta_13_min': process_delta_13_min,
        'delta_13_max': process_delta_13_max,
        'delta_13_range': process_delta_13_range,
        'delta_13_mean': process_delta_13_mean,
        'delta_13_std': process_delta_13_std,
        'delta_13_median': process_delta_13_median,
        'delta_13_median_min': process_delta_13_median_min,
        'delta_13_median_max': process_delta_13_median_max,

        #Delta 14
        'delta_14_min': process_delta_14_min,
        'delta_14_max': process_delta_14_max,
        'delta_14_range': process_delta_14_range,
        'delta_14_mean': process_delta_14_mean,
        'delta_14_std': process_delta_14_std,
        'delta_14_median': process_delta_14_median,
        'delta_14_median_min': process_delta_14_median_min,
        'delta_14_median_max': process_delta_14_median_max,

        #Delta 15
        'delta_15_min': process_delta_15_min,
        'delta_15_max': process_delta_15_max,
        'delta_15_range': process_delta_15_range,
        'delta_15_mean': process_delta_15_mean,
        'delta_15_std': process_delta_15_std,
        'delta_15_median': process_delta_15_median,
        'delta_15_median_min': process_delta_15_median_min,
        'delta_15_median_max': process_delta_15_median_max,

        #Delta 16
        'delta_16_min': process_delta_16_min,
        'delta_16_max': process_delta_16_max,
        'delta_16_range': process_delta_16_range,
        'delta_16_mean': process_delta_16_mean,
        'delta_16_std': process_delta_16_std,
        'delta_16_median': process_delta_16_median,
        'delta_16_median_min': process_delta_16_median_min,
        'delta_16_median_max': process_delta_16_median_max,

        #Delta 17
        'delta_17_min': process_delta_17_min,
        'delta_17_max': process_delta_17_max,
        'delta_17_range': process_delta_17_range,
        'delta_17_mean': process_delta_17_mean,
        'delta_17_std': process_delta_17_std,
        'delta_17_median': process_delta_17_median,
        'delta_17_median_min': process_delta_17_median_min,
        'delta_17_median_max': process_delta_17_median_max,

        #Delta 18
        'delta_18_min': process_delta_18_min,
        'delta_18_max': process_delta_18_max,
        'delta_18_range': process_delta_18_range,
        'delta_18_mean': process_delta_18_mean,
        'delta_18_std': process_delta_18_std,
        'delta_18_median': process_delta_18_median,
        'delta_18_median_min': process_delta_18_median_min,
        'delta_18_median_max': process_delta_18_median_max,

        #Delta 19
        'delta_19_min': process_delta_19_min,
        'delta_19_max': process_delta_19_max,
        'delta_19_range': process_delta_19_range,
        'delta_19_mean': process_delta_19_mean,
        'delta_19_std': process_delta_19_std,
        'delta_19_median': process_delta_19_median,
        'delta_19_median_min': process_delta_19_median_min,
        'delta_19_median_max': process_delta_19_median_max,
        }
    return processing_functions

#amplitude
def process_amp_min(audio, sr):
    return np.min(librosa.amplitude_to_db(audio))
def process_amp_max(audio, sr):
    return np.max(librosa.amplitude_to_db(audio))
def process_amp_range(audio, sr):
    return np.max(librosa.amplitude_to_db(audio)) - np.min(librosa.amplitude_to_db(audio))
def process_amp_mean(audio, sr):
    return np.mean(librosa.amplitude_to_db(audio))
def process_amp_std(audio, sr):
    return np.std(librosa.amplitude_to_db(audio))
def process_amp_median(audio, sr):
    return np.median(librosa.amplitude_to_db(audio))
def process_amp_median_min(audio, sr):
    return abs(np.min(librosa.amplitude_to_db(audio)) - np.median(librosa.amplitude_to_db(audio)))
def process_amp_median_max(audio, sr):
    return abs(np.max(librosa.amplitude_to_db(audio)) - np.median(librosa.amplitude_to_db(audio)))

#root-mean-square value for each frame
def process_rms_min(audio, sr):
    return np.min(librosa.feature.rms(y=audio)[0,:])
def process_rms_max(audio, sr):
    return np.max(librosa.feature.rms(y=audio)[0,:])
def process_rms_range(audio, sr):
    return np.max(librosa.feature.rms(y=audio)[0,:]) - np.min(librosa.feature.rms(y=audio)[0,:])
def process_rms_mean(audio, sr):
    return np.mean(librosa.feature.rms(y=audio)[0,:])
def process_rms_std(audio, sr):
    return np.std(librosa.feature.rms(y=audio)[0,:])
def process_rms_median(audio, sr):
    return np.median(librosa.feature.rms(y=audio)[0,:])
def process_rms_median_min(audio, sr):
    return abs(np.min(librosa.feature.rms(y=audio)[0,:]) - np.median(librosa.feature.rms(y=audio)[0,:]))
def process_rms_median_max(audio, sr):
    return abs(np.max(librosa.feature.rms(y=audio)[0,:]) - np.median(librosa.feature.rms(y=audio)[0,:]))

#polynomial coefficients for each frame (order 0)
def process_poly_features_min(audio, sr):
    return np.min(librosa.feature.poly_features(y=audio, sr=sr, order=0)[0,:])
def process_poly_features_max(audio, sr):
    return np.max(librosa.feature.poly_features(y=audio, sr=sr, order=0)[0,:])
def process_poly_features_range(audio, sr):
    return np.max(librosa.feature.poly_features(y=audio, sr=sr, order=0)[0,:]) - np.min(librosa.feature.poly_features(y=audio, sr=sr, order=0)[0,:])
def process_poly_features_mean(audio, sr):
    return np.mean(librosa.feature.poly_features(y=audio, sr=sr, order=0)[0,:])
def process_poly_features_std(audio, sr):
    return np.std(librosa.feature.poly_features(y=audio, sr=sr, order=0)[0,:])
def process_poly_features_median(audio, sr):
    return np.median(librosa.feature.poly_features(y=audio, sr=sr, order=0)[0,:])
def process_poly_features_median_min(audio, sr):
    return abs(np.min(librosa.feature.poly_features(y=audio, sr=sr, order=0)[0,:]) - np.median(librosa.feature.poly_features(y=audio, sr=sr, order=0)[0,:]))
def process_poly_features_median_max(audio, sr):
    return abs(np.max(librosa.feature.poly_features(y=audio, sr=sr, order=0)[0,:]) - np.median(librosa.feature.poly_features(y=audio, sr=sr, order=0)[0,:]))

#spectral bandwidth
def process_spectral_bandwidth_min(audio, sr):
    return np.min(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0,:])
def process_spectral_bandwidth_max(audio, sr):
    return np.max(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0,:])
def process_spectral_bandwidth_range(audio, sr):
    return np.max(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0,:]) - np.min(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0,:])
def process_spectral_bandwidth_mean(audio, sr):
    return np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0,:])
def process_spectral_bandwidth_std(audio, sr):
    return np.std(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0,:])
def process_spectral_bandwidth_median(audio, sr):
    return np.median(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0,:])
def process_spectral_bandwidth_median_min(audio, sr):
    return abs(np.min(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0,:]) - np.median(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0,:]))
def process_spectral_bandwidth_median_max(audio, sr):
    return abs(np.max(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0,:]) - np.median(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0,:]))

#spectral centroid
def process_spectral_centroid_min(audio, sr):
    return np.min(librosa.feature.spectral_centroid(y=audio, sr=sr)[0,:])
def process_spectral_centroid_max(audio, sr):
    return np.max(librosa.feature.spectral_centroid(y=audio, sr=sr)[0,:])
def process_spectral_centroid_range(audio, sr):
    return np.max(librosa.feature.spectral_centroid(y=audio, sr=sr)[0,:]) - np.min(librosa.feature.spectral_centroid(y=audio, sr=sr)[0,:])
def process_spectral_centroid_mean(audio, sr):
    return np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0,:])
def process_spectral_centroid_std(audio, sr):
    return np.std(librosa.feature.spectral_centroid(y=audio, sr=sr)[0,:])
def process_spectral_centroid_median(audio, sr):
    return np.median(librosa.feature.spectral_centroid(y=audio, sr=sr)[0,:])
def process_spectral_centroid_median_min(audio, sr):
    return abs(np.min(librosa.feature.spectral_centroid(y=audio, sr=sr)[0,:]) - np.median(librosa.feature.spectral_centroid(y=audio, sr=sr)[0,:]))
def process_spectral_centroid_median_max(audio, sr):
    return abs(np.max(librosa.feature.spectral_centroid(y=audio, sr=sr)[0,:]) - np.median(librosa.feature.spectral_centroid(y=audio, sr=sr)[0,:]))

#spectral flatness
def process_spectral_flatness_min(audio, sr):
    return np.min(librosa.feature.spectral_flatness(y=audio)[0,:])
def process_spectral_flatness_max(audio, sr):
    return np.max(librosa.feature.spectral_flatness(y=audio)[0,:])
def process_spectral_flatness_range(audio, sr):
    return np.max(librosa.feature.spectral_flatness(y=audio)[0,:]) - np.min(librosa.feature.spectral_flatness(y=audio)[0,:])
def process_spectral_flatness_mean(audio, sr):
    return np.mean(librosa.feature.spectral_flatness(y=audio)[0,:])
def process_spectral_flatness_std(audio, sr):
    return np.std(librosa.feature.spectral_flatness(y=audio)[0,:])
def process_spectral_flatness_median(audio, sr):
    return np.median(librosa.feature.spectral_flatness(y=audio)[0,:])
def process_spectral_flatness_median_min(audio, sr):
    return abs(np.min(librosa.feature.spectral_flatness(y=audio)[0,:]) - np.median(librosa.feature.spectral_flatness(y=audio)[0,:]))
def process_spectral_flatness_median_max(audio, sr):
    return abs(np.max(librosa.feature.spectral_flatness(y=audio)[0,:]) - np.median(librosa.feature.spectral_flatness(y=audio)[0,:]))

#spectral rolloff
def process_spectral_rolloff_min(audio, sr):
    return np.min(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0,:])
def process_spectral_rolloff_max(audio, sr):
    return np.max(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0,:])
def process_spectral_rolloff_range(audio, sr):
    return np.max(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0,:]) - np.min(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0,:])
def process_spectral_rolloff_mean(audio, sr):
    return np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0,:])
def process_spectral_rolloff_std(audio, sr):
    return np.std(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0,:])
def process_spectral_rolloff_median(audio, sr):
    return np.median(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0,:])
def process_spectral_rolloff_median_min(audio, sr):
    return abs(np.min(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0,:]) - np.median(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0,:]))
def process_spectral_rolloff_median_max(audio, sr):
    return abs(np.max(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0,:]) - np.median(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0,:]))

#zero crossing rate
def process_zero_crossing_rate_min(audio, sr):
    return np.min(librosa.feature.zero_crossing_rate(y=audio)[0,:])
def process_zero_crossing_rate_max(audio, sr):
    return np.max(librosa.feature.zero_crossing_rate(y=audio)[0,:])
def process_zero_crossing_rate_range(audio, sr):
    return np.max(librosa.feature.zero_crossing_rate(y=audio)[0,:]) - np.min(librosa.feature.zero_crossing_rate(y=audio)[0,:])
def process_zero_crossing_rate_mean(audio, sr):
    return np.mean(librosa.feature.zero_crossing_rate(y=audio)[0,:])
def process_zero_crossing_rate_std(audio, sr):
    return np.std(librosa.feature.zero_crossing_rate(y=audio)[0,:])
def process_zero_crossing_rate_median(audio, sr):
    return np.median(librosa.feature.zero_crossing_rate(y=audio)[0,:])
def process_zero_crossing_rate_median_min(audio, sr):
    return abs(np.min(librosa.feature.zero_crossing_rate(y=audio)[0,:]) - np.median(librosa.feature.zero_crossing_rate(y=audio)[0,:]))
def process_zero_crossing_rate_median_max(audio, sr):
    return abs(np.max(librosa.feature.zero_crossing_rate(y=audio)[0,:]) - np.median(librosa.feature.zero_crossing_rate(y=audio)[0,:]))

#tempo
def process_tempo(audio, sr):
    # return librosa.beat.tempo(y=audio, sr=sr)[0]
    return librosa.feature.tempo(y=audio, sr=sr)[0]


#chromagram (pitch) bin 0
def process_chroma_cens_0_min(audio, sr):
    return np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[0,:])
def process_chroma_cens_0_max(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[0,:])
def process_chroma_cens_0_range(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[0,:]) - np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[0,:])
def process_chroma_cens_0_mean(audio, sr):
    return np.mean(librosa.feature.chroma_cens(y=audio, sr=sr)[0,:])
def process_chroma_cens_0_std(audio, sr):
    return np.std(librosa.feature.chroma_cens(y=audio, sr=sr)[0,:])
def process_chroma_cens_0_median(audio, sr):
    return np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[0,:])
def process_chroma_cens_0_median_min(audio, sr):
    return abs(np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[0,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[0,:]))
def process_chroma_cens_0_median_max(audio, sr):
    return abs(np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[0,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[0,:]))

#chromagram (pitch) bin 1
def process_chroma_cens_1_min(audio, sr):
    return np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[1,:])
def process_chroma_cens_1_max(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[1,:])
def process_chroma_cens_1_range(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[1,:]) - np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[1,:])
def process_chroma_cens_1_mean(audio, sr):
    return np.mean(librosa.feature.chroma_cens(y=audio, sr=sr)[1,:])
def process_chroma_cens_1_std(audio, sr):
    return np.std(librosa.feature.chroma_cens(y=audio, sr=sr)[1,:])
def process_chroma_cens_1_median(audio, sr):
    return np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[1,:])
def process_chroma_cens_1_median_min(audio, sr):
    return abs(np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[1,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[1,:]))
def process_chroma_cens_1_median_max(audio, sr):
    return abs(np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[1,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[1,:]))

#chromagram (pitch) bin 2
def process_chroma_cens_2_min(audio, sr):
    return np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[2,:])
def process_chroma_cens_2_max(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[2,:])
def process_chroma_cens_2_range(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[2,:]) - np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[1,:])
def process_chroma_cens_2_mean(audio, sr):
    return np.mean(librosa.feature.chroma_cens(y=audio, sr=sr)[2,:])
def process_chroma_cens_2_std(audio, sr):
    return np.std(librosa.feature.chroma_cens(y=audio, sr=sr)[2,:])
def process_chroma_cens_2_median(audio, sr):
    return np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[2,:])
def process_chroma_cens_2_median_min(audio, sr):
    return abs(np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[2,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[2,:]))
def process_chroma_cens_2_median_max(audio, sr):
    return abs(np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[2,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[2,:]))

#chromagram (pitch) bin 3
def process_chroma_cens_3_min(audio, sr):
    return np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[3,:])
def process_chroma_cens_3_max(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[3,:])
def process_chroma_cens_3_range(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[3,:]) - np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[3,:])
def process_chroma_cens_3_mean(audio, sr):
    return np.mean(librosa.feature.chroma_cens(y=audio, sr=sr)[3,:])
def process_chroma_cens_3_std(audio, sr):
    return np.std(librosa.feature.chroma_cens(y=audio, sr=sr)[3,:])
def process_chroma_cens_3_median(audio, sr):
    return np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[3,:])
def process_chroma_cens_3_median_min(audio, sr):
    return abs(np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[3,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[3,:]))
def process_chroma_cens_3_median_max(audio, sr):
    return abs(np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[3,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[3,:]))

#chromagram (pitch) bin 4
def process_chroma_cens_4_min(audio, sr):
    return np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[4,:])
def process_chroma_cens_4_max(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[4,:])
def process_chroma_cens_4_range(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[4,:]) - np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[4,:])
def process_chroma_cens_4_mean(audio, sr):
    return np.mean(librosa.feature.chroma_cens(y=audio, sr=sr)[4,:])
def process_chroma_cens_4_std(audio, sr):
    return np.std(librosa.feature.chroma_cens(y=audio, sr=sr)[4,:])
def process_chroma_cens_4_median(audio, sr):
    return np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[4,:])
def process_chroma_cens_4_median_min(audio, sr):
    return abs(np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[4,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[4,:]))
def process_chroma_cens_4_median_max(audio, sr):
    return abs(np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[4,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[4,:]))

#chromagram (pitch) bin 5
def process_chroma_cens_5_min(audio, sr):
    return np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[5,:])
def process_chroma_cens_5_max(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[5,:])
def process_chroma_cens_5_range(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[5,:]) - np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[5,:])
def process_chroma_cens_5_mean(audio, sr):
    return np.mean(librosa.feature.chroma_cens(y=audio, sr=sr)[5,:])
def process_chroma_cens_5_std(audio, sr):
    return np.std(librosa.feature.chroma_cens(y=audio, sr=sr)[5,:])
def process_chroma_cens_5_median(audio, sr):
    return np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[5,:])
def process_chroma_cens_5_median_min(audio, sr):
    return abs(np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[5,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[5,:]))
def process_chroma_cens_5_median_max(audio, sr):
    return abs(np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[5,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[5,:]))

#chromagram (pitch) bin 6
def process_chroma_cens_6_min(audio, sr):
    return np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[6,:])
def process_chroma_cens_6_max(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[6,:])
def process_chroma_cens_6_range(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[6,:]) - np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[6,:])
def process_chroma_cens_6_mean(audio, sr):
    return np.mean(librosa.feature.chroma_cens(y=audio, sr=sr)[6,:])
def process_chroma_cens_6_std(audio, sr):
    return np.std(librosa.feature.chroma_cens(y=audio, sr=sr)[6,:])
def process_chroma_cens_6_median(audio, sr):
    return np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[6,:])
def process_chroma_cens_6_median_min(audio, sr):
    return abs(np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[6,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[6,:]))
def process_chroma_cens_6_median_max(audio, sr):
    return abs(np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[6,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[6,:]))

#chromagram (pitch) bin 7
def process_chroma_cens_7_min(audio, sr):
    return np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[7,:])
def process_chroma_cens_7_max(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[7,:])
def process_chroma_cens_7_range(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[7,:]) - np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[7,:])
def process_chroma_cens_7_mean(audio, sr):
    return np.mean(librosa.feature.chroma_cens(y=audio, sr=sr)[7,:])
def process_chroma_cens_7_std(audio, sr):
    return np.std(librosa.feature.chroma_cens(y=audio, sr=sr)[7,:])
def process_chroma_cens_7_median(audio, sr):
    return np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[7,:])
def process_chroma_cens_7_median_min(audio, sr):
    return abs(np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[7,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[7,:]))
def process_chroma_cens_7_median_max(audio, sr):
    return abs(np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[7,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[7,:]))

#chromagram (pitch) bin 8
def process_chroma_cens_8_min(audio, sr):
    return np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[8,:])
def process_chroma_cens_8_max(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[8,:])
def process_chroma_cens_8_range(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[8,:]) - np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[8,:])
def process_chroma_cens_8_mean(audio, sr):
    return np.mean(librosa.feature.chroma_cens(y=audio, sr=sr)[8,:])
def process_chroma_cens_8_std(audio, sr):
    return np.std(librosa.feature.chroma_cens(y=audio, sr=sr)[8,:])
def process_chroma_cens_8_median(audio, sr):
    return np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[8,:])
def process_chroma_cens_8_median_min(audio, sr):
    return abs(np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[8,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[8,:]))
def process_chroma_cens_8_median_max(audio, sr):
    return abs(np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[8,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[8,:]))

#chromagram (pitch) bin 9
def process_chroma_cens_9_min(audio, sr):
    return np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[9,:])
def process_chroma_cens_9_max(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[9,:])
def process_chroma_cens_9_range(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[9,:]) - np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[9,:])
def process_chroma_cens_9_mean(audio, sr):
    return np.mean(librosa.feature.chroma_cens(y=audio, sr=sr)[9,:])
def process_chroma_cens_9_std(audio, sr):
    return np.std(librosa.feature.chroma_cens(y=audio, sr=sr)[9,:])
def process_chroma_cens_9_median(audio, sr):
    return np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[9,:])
def process_chroma_cens_9_median_min(audio, sr):
    return abs(np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[9,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[9,:]))
def process_chroma_cens_9_median_max(audio, sr):
    return abs(np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[9,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[9,:]))

#chromagram (pitch) bin 10
def process_chroma_cens_10_min(audio, sr):
    return np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[10,:])
def process_chroma_cens_10_max(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[10,:])
def process_chroma_cens_10_range(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[10,:]) - np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[10,:])
def process_chroma_cens_10_mean(audio, sr):
    return np.mean(librosa.feature.chroma_cens(y=audio, sr=sr)[10,:])
def process_chroma_cens_10_std(audio, sr):
    return np.std(librosa.feature.chroma_cens(y=audio, sr=sr)[10,:])
def process_chroma_cens_10_median(audio, sr):
    return np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[10,:])
def process_chroma_cens_10_median_min(audio, sr):
    return abs(np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[10,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[10,:]))
def process_chroma_cens_10_median_max(audio, sr):
    return abs(np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[10,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[10,:]))

#chromagram (pitch) bin 11
def process_chroma_cens_11_min(audio, sr):
    return np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[11,:])
def process_chroma_cens_11_max(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[11,:])
def process_chroma_cens_11_range(audio, sr):
    return np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[11,:]) - np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[11,:])
def process_chroma_cens_11_mean(audio, sr):
    return np.mean(librosa.feature.chroma_cens(y=audio, sr=sr)[11,:])
def process_chroma_cens_11_std(audio, sr):
    return np.std(librosa.feature.chroma_cens(y=audio, sr=sr)[11,:])
def process_chroma_cens_11_median(audio, sr):
    return np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[11,:])
def process_chroma_cens_11_median_min(audio, sr):
    return abs(np.min(librosa.feature.chroma_cens(y=audio, sr=sr)[11,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[11,:]))
def process_chroma_cens_11_median_max(audio, sr):
    return abs(np.max(librosa.feature.chroma_cens(y=audio, sr=sr)[11,:]) - np.median(librosa.feature.chroma_cens(y=audio, sr=sr)[11,:]))


#MFCC 0
def process_mfcc_0_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[0,:])
def process_mfcc_0_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[0,:])
def process_mfcc_0_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[0,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[0,:])
def process_mfcc_0_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[0,:])
def process_mfcc_0_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[0,:])
def process_mfcc_0_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[0,:])
def process_mfcc_0_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[0,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[0,:]))
def process_mfcc_0_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[0,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[0,:]))

#MFCC 1
def process_mfcc_1_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[1,:])
def process_mfcc_1_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[1,:])
def process_mfcc_1_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[1,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[1,:])
def process_mfcc_1_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[1,:])
def process_mfcc_1_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[1,:])
def process_mfcc_1_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[1,:])
def process_mfcc_1_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[1,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[1,:]))
def process_mfcc_1_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[1,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[1,:]))

#MFCC 2
def process_mfcc_2_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[2,:])
def process_mfcc_2_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[2,:])
def process_mfcc_2_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[2,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[2,:])
def process_mfcc_2_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[2,:])
def process_mfcc_2_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[2,:])
def process_mfcc_2_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[2,:])
def process_mfcc_2_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[2,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[2,:]))
def process_mfcc_2_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[2,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[2,:]))

#MFCC 3
def process_mfcc_3_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[3,:])
def process_mfcc_3_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[3,:])
def process_mfcc_3_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[3,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[3,:])
def process_mfcc_3_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[3,:])
def process_mfcc_3_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[3,:])
def process_mfcc_3_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[3,:])
def process_mfcc_3_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[3,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[3,:]))
def process_mfcc_3_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[3,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[3,:]))

#MFCC 4
def process_mfcc_4_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[4,:])
def process_mfcc_4_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[4,:])
def process_mfcc_4_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[4,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[4,:])
def process_mfcc_4_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[4,:])
def process_mfcc_4_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[4,:])
def process_mfcc_4_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[4,:])
def process_mfcc_4_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[4,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[4,:]))
def process_mfcc_4_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[4,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[4,:]))

#MFCC 5
def process_mfcc_5_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[5,:])
def process_mfcc_5_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[5,:])
def process_mfcc_5_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[5,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[5,:])
def process_mfcc_5_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[5,:])
def process_mfcc_5_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[5,:])
def process_mfcc_5_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[5,:])
def process_mfcc_5_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[5,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[5,:]))
def process_mfcc_5_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[5,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[5,:]))

#MFCC 6
def process_mfcc_6_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[6,:])
def process_mfcc_6_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[6,:])
def process_mfcc_6_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[6,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[6,:])
def process_mfcc_6_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[6,:])
def process_mfcc_6_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[6,:])
def process_mfcc_6_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[6,:])
def process_mfcc_6_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[6,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[6,:]))
def process_mfcc_6_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[6,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[6,:]))

#MFCC 7
def process_mfcc_7_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[7,:])
def process_mfcc_7_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[7,:])
def process_mfcc_7_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[7,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[7,:])
def process_mfcc_7_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[7,:])
def process_mfcc_7_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[7,:])
def process_mfcc_7_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[7,:])
def process_mfcc_7_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[7,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[7,:]))
def process_mfcc_7_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[7,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[7,:]))

#MFCC 8
def process_mfcc_8_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[8,:])
def process_mfcc_8_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[8,:])
def process_mfcc_8_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[8,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[8,:])
def process_mfcc_8_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[8,:])
def process_mfcc_8_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[8,:])
def process_mfcc_8_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[8,:])
def process_mfcc_8_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[8,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[8,:]))
def process_mfcc_8_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[8,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[8,:]))

#MFCC 9
def process_mfcc_9_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[9,:])
def process_mfcc_9_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[9,:])
def process_mfcc_9_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[9,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[9,:])
def process_mfcc_9_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[9,:])
def process_mfcc_9_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[9,:])
def process_mfcc_9_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[9,:])
def process_mfcc_9_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[9,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[9,:]))
def process_mfcc_9_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[9,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[9,:]))

#MFCC 10
def process_mfcc_10_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[10,:])
def process_mfcc_10_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[10,:])
def process_mfcc_10_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[10,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[10,:])
def process_mfcc_10_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[10,:])
def process_mfcc_10_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[10,:])
def process_mfcc_10_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[10,:])
def process_mfcc_10_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[10,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[10,:]))
def process_mfcc_10_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[10,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[10,:]))

#MFCC 11
def process_mfcc_11_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[11,:])
def process_mfcc_11_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[11,:])
def process_mfcc_11_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[11,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[11,:])
def process_mfcc_11_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[11,:])
def process_mfcc_11_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[11,:])
def process_mfcc_11_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[11,:])
def process_mfcc_11_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[11,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[11,:]))
def process_mfcc_11_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[11,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[11,:]))

#MFCC 12
def process_mfcc_12_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[12,:])
def process_mfcc_12_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[12,:])
def process_mfcc_12_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[12,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[12,:])
def process_mfcc_12_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[12,:])
def process_mfcc_12_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[12,:])
def process_mfcc_12_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[12,:])
def process_mfcc_12_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[12,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[12,:]))
def process_mfcc_12_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[12,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[12,:]))

#MFCC 13
def process_mfcc_13_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[13,:])
def process_mfcc_13_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[13,:])
def process_mfcc_13_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[13,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[13,:])
def process_mfcc_13_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[13,:])
def process_mfcc_13_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[13,:])
def process_mfcc_13_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[13,:])
def process_mfcc_13_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[13,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[13,:]))
def process_mfcc_13_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[13,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[13,:]))

#MFCC 14
def process_mfcc_14_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[14,:])
def process_mfcc_14_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[14,:])
def process_mfcc_14_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[14,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[14,:])
def process_mfcc_14_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[14,:])
def process_mfcc_14_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[14,:])
def process_mfcc_14_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[14,:])
def process_mfcc_14_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[14,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[14,:]))
def process_mfcc_14_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[14,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[14,:]))

#MFCC 15
def process_mfcc_15_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[15,:])
def process_mfcc_15_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[15,:])
def process_mfcc_15_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[15,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[15,:])
def process_mfcc_15_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[15,:])
def process_mfcc_15_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[15,:])
def process_mfcc_15_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[15,:])
def process_mfcc_15_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[15,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[15,:]))
def process_mfcc_15_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[15,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[15,:]))

#MFCC 16
def process_mfcc_16_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[16,:])
def process_mfcc_16_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[16,:])
def process_mfcc_16_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[16,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[16,:])
def process_mfcc_16_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[16,:])
def process_mfcc_16_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[16,:])
def process_mfcc_16_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[16,:])
def process_mfcc_16_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[16,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[16,:]))
def process_mfcc_16_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[16,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[16,:]))

#MFCC 17
def process_mfcc_17_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[17,:])
def process_mfcc_17_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[17,:])
def process_mfcc_17_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[17,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[15,:])
def process_mfcc_17_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[17,:])
def process_mfcc_17_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[17,:])
def process_mfcc_17_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[17,:])
def process_mfcc_17_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[17,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[17,:]))
def process_mfcc_17_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[17,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[17,:]))

#MFCC 18
def process_mfcc_18_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[18,:])
def process_mfcc_18_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[18,:])
def process_mfcc_18_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[18,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[18,:])
def process_mfcc_18_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[18,:])
def process_mfcc_18_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[18,:])
def process_mfcc_18_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[18,:])
def process_mfcc_18_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[18,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[18,:]))
def process_mfcc_18_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[18,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[18,:]))

#MFCC 19
def process_mfcc_19_min(audio, sr):
    return np.min(librosa.feature.mfcc(y=audio, sr=sr)[19,:])
def process_mfcc_19_max(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[19,:])
def process_mfcc_19_range(audio, sr):
    return np.max(librosa.feature.mfcc(y=audio, sr=sr)[19,:]) - np.min(librosa.feature.mfcc(y=audio, sr=sr)[19,:])
def process_mfcc_19_mean(audio, sr):
    return np.mean(librosa.feature.mfcc(y=audio, sr=sr)[19,:])
def process_mfcc_19_std(audio, sr):
    return np.std(librosa.feature.mfcc(y=audio, sr=sr)[19,:])
def process_mfcc_19_median(audio, sr):
    return np.median(librosa.feature.mfcc(y=audio, sr=sr)[19,:])
def process_mfcc_19_median_min(audio, sr):
    return abs(np.min(librosa.feature.mfcc(y=audio, sr=sr)[19,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[19,:]))
def process_mfcc_19_median_max(audio, sr):
    return abs(np.max(librosa.feature.mfcc(y=audio, sr=sr)[19,:]) - np.median(librosa.feature.mfcc(y=audio, sr=sr)[19,:]))

#Delta 0
def process_delta_0_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[0,:])
def process_delta_0_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[0,:])
def process_delta_0_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[0,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[0,:])
def process_delta_0_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[0,:])
def process_delta_0_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[0,:])
def process_delta_0_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[0,:])
def process_delta_0_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[0,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[0,:]))
def process_delta_0_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[0,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[0,:]))

#Delta 1
def process_delta_1_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[1,:])
def process_delta_1_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[1,:])
def process_delta_1_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[1,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[1,:])
def process_delta_1_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[1,:])
def process_delta_1_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[1,:])
def process_delta_1_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[1,:])
def process_delta_1_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[1,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[1,:]))
def process_delta_1_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[1,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[1,:]))

#Delta 2
def process_delta_2_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[2,:])
def process_delta_2_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[2,:])
def process_delta_2_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[2,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[2,:])
def process_delta_2_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[2,:])
def process_delta_2_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[2,:])
def process_delta_2_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[2,:])
def process_delta_2_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[2,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[2,:]))
def process_delta_2_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[2,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[2,:]))

#Delta 3
def process_delta_3_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[3,:])
def process_delta_3_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[3,:])
def process_delta_3_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[3,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[3,:])
def process_delta_3_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[3,:])
def process_delta_3_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[3,:])
def process_delta_3_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[3,:])
def process_delta_3_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[3,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[3,:]))
def process_delta_3_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[3,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[3,:]))

#Delta 4
def process_delta_4_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[4,:])
def process_delta_4_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[4,:])
def process_delta_4_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[4,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[4,:])
def process_delta_4_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[4,:])
def process_delta_4_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[4,:])
def process_delta_4_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[4,:])
def process_delta_4_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[4,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[4,:]))
def process_delta_4_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[4,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[4,:]))

#Delta 5
def process_delta_5_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[5,:])
def process_delta_5_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[5,:])
def process_delta_5_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[5,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[5,:])
def process_delta_5_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[5,:])
def process_delta_5_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[5,:])
def process_delta_5_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[5,:])
def process_delta_5_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[5,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[5,:]))
def process_delta_5_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[5,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[5,:]))

#Delta 6
def process_delta_6_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[6,:])
def process_delta_6_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[6,:])
def process_delta_6_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[6,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[6,:])
def process_delta_6_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[6,:])
def process_delta_6_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[6,:])
def process_delta_6_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[6,:])
def process_delta_6_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[6,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[6,:]))
def process_delta_6_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[6,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[6,:]))

#Delta 7
def process_delta_7_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[7,:])
def process_delta_7_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[7,:])
def process_delta_7_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[7,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[7,:])
def process_delta_7_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[7,:])
def process_delta_7_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[7,:])
def process_delta_7_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[7,:])
def process_delta_7_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[7,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[7,:]))
def process_delta_7_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[7,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[7,:]))

#Delta 8
def process_delta_8_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[8,:])
def process_delta_8_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[8,:])
def process_delta_8_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[8,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[8,:])
def process_delta_8_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[8,:])
def process_delta_8_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[8,:])
def process_delta_8_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[8,:])
def process_delta_8_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[8,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[8,:]))
def process_delta_8_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[8,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[8,:]))

#Delta 9
def process_delta_9_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[9,:])
def process_delta_9_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[9,:])
def process_delta_9_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[9,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[9,:])
def process_delta_9_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[9,:])
def process_delta_9_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[9,:])
def process_delta_9_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[9,:])
def process_delta_9_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[9,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[9,:]))
def process_delta_9_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[9,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[9,:]))

#Delta 10
def process_delta_10_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[10,:])
def process_delta_10_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[10,:])
def process_delta_10_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[10,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[10,:])
def process_delta_10_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[10,:])
def process_delta_10_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[10,:])
def process_delta_10_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[10,:])
def process_delta_10_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[10,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[10,:]))
def process_delta_10_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[10,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[10,:]))

#Delta 11
def process_delta_11_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[11,:])
def process_delta_11_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[11,:])
def process_delta_11_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[11,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[11,:])
def process_delta_11_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[11,:])
def process_delta_11_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[11,:])
def process_delta_11_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[11,:])
def process_delta_11_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[11,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[11,:]))
def process_delta_11_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[11,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[11,:]))

#Delta 12
def process_delta_12_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[12,:])
def process_delta_12_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[12,:])
def process_delta_12_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[12,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[12,:])
def process_delta_12_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[12,:])
def process_delta_12_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[12,:])
def process_delta_12_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[12,:])
def process_delta_12_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[12,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[12,:]))
def process_delta_12_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[12,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[12,:]))

#Delta 13
def process_delta_13_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[13,:])
def process_delta_13_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[13,:])
def process_delta_13_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[13,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[13,:])
def process_delta_13_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[13,:])
def process_delta_13_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[13,:])
def process_delta_13_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[13,:])
def process_delta_13_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[13,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[13,:]))
def process_delta_13_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[13,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[13,:]))

#Delta 14
def process_delta_14_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[14,:])
def process_delta_14_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[14,:])
def process_delta_14_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[14,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[14,:])
def process_delta_14_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[14,:])
def process_delta_14_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[14,:])
def process_delta_14_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[14,:])
def process_delta_14_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[14,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[14,:]))
def process_delta_14_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[14,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[14,:]))

#Delta 15
def process_delta_15_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[15,:])
def process_delta_15_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[15,:])
def process_delta_15_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[15,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[15,:])
def process_delta_15_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[15,:])
def process_delta_15_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[15,:])
def process_delta_15_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[15,:])
def process_delta_15_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[15,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[15,:]))
def process_delta_15_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[15,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[15,:]))

#Delta 16
def process_delta_16_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[16,:])
def process_delta_16_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[16,:])
def process_delta_16_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[16,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[16,:])
def process_delta_16_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[16,:])
def process_delta_16_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[16,:])
def process_delta_16_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[16,:])
def process_delta_16_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[16,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[16,:]))
def process_delta_16_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[16,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[16,:]))

#Delta 17
def process_delta_17_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[17,:])
def process_delta_17_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[17,:])
def process_delta_17_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[17,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[17,:])
def process_delta_17_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[17,:])
def process_delta_17_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[17,:])
def process_delta_17_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[17,:])
def process_delta_17_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[17,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[17,:]))
def process_delta_17_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[17,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[17,:]))

#Delta 18
def process_delta_18_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[18,:])
def process_delta_18_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[18,:])
def process_delta_18_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[18,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[18,:])
def process_delta_18_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[18,:])
def process_delta_18_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[18,:])
def process_delta_18_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[18,:])
def process_delta_18_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[18,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[18,:]))
def process_delta_18_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[18,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[18,:]))

#Delta 19
def process_delta_19_min(audio, sr):
    return np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[19,:])
def process_delta_19_max(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[19,:])
def process_delta_19_range(audio, sr):
    return np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[19,:]) - np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[19,:])
def process_delta_19_mean(audio, sr):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[19,:])
def process_delta_19_std(audio, sr):
    return np.std(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[19,:])
def process_delta_19_median(audio, sr):
    return np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[19,:])
def process_delta_19_median_min(audio, sr):
    return abs(np.min(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[19,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[19,:]))
def process_delta_19_median_max(audio, sr):
    return abs(np.max(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[19,:]) - np.median(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr))[19,:]))
