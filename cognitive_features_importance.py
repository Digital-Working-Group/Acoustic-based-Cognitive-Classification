""" We employed Random Forest Regression with an ensemble of 100 decision tree estimators to calculate the importance of voice features. 
The importance values were then normalized to the interval [0, 1]. 
Finally, the features and their corresponding importance scores were saved as a CSV file for output.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def calculate_features_importance(**kwargs):
    """
    calculate features importance;
    """
    voice_features = kwargs.get('voice_features')
    extract_features_csv_out = kwargs.get('extract_features_csv_out')
    features_importance_csv_out = kwargs.get('features_importance_csv_out')
    df = pd.read_csv(extract_features_csv_out, sep=',', usecols=voice_features)
    df_arr = df.to_numpy()
    df_labels = pd.read_csv(extract_features_csv_out, sep=',', usecols=['DX'])
    df_fhs_labels_arr = df_labels.to_numpy().ravel()
    clf = RandomForestRegressor(n_estimators=100)
    clf.fit(df_arr, df_fhs_labels_arr)
    importance = clf.feature_importances_
    weights_df = pd.DataFrame(columns=["Feature", "Importance", "Importance_norm"])
    weights_df['Feature'] = voice_features
    weights_df['Importance'] = importance
    weights_sum = weights_df['Importance'].sum()
    weights_df['Importance_norm'] = weights_df['Importance'] / weights_sum #normalize the importance of features over interval [0, 1]
    final_weights_df = weights_df.sort_values(by=['Importance_norm'], ascending=False)
    final_weights_df.to_csv(features_importance_csv_out, sep=',', index=False)
    print(f'wrote {features_importance_csv_out}')
