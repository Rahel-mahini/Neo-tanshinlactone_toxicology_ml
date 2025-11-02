# -*- coding: utf-8 -*-
"""
Feature selection using Gentic Algorithm
"""
# step05_feature_selection.py
from sklearn.linear_model import MultiTaskLasso
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np

def run_mtlasso_rfe_per_target(X, Y, n_features_list=[1, 2, 3, 4]):
    """
    Perform joint (multi-task) feature selection using MultiTaskLasso with RFE.
    Returns a dictionary: {n_features: list of top features}
    """

    best_features_dict = {}

    # Ensure Y is 2D (multi-task: each column is one target)
    if isinstance(Y, pd.Series):
        Y = Y.to_frame()
    elif isinstance(Y, np.ndarray) and Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    print("X info:")
    print(X.info())
    print("Y info:")
    print(Y.info())
    print("X head:")
    print(X.head())
    print("Y head:")
    print(Y.head())

    # Iterate over the number of features to select
    for n_features in n_features_list:
        # Define the MultiTaskLasso estimator
        estimator = MultiTaskLasso(alpha=0.01, random_state=42)

        # Define RFE wrapper
        selector = RFE(estimator, n_features_to_select=n_features)

        # Fit RFE on the training data
        selector.fit(X, Y)

        # Extract selected features
        selected_features = X.columns[selector.support_].tolist()
        best_features_dict[n_features] = selected_features

        print(f"Selected top {n_features} features: {selected_features}")

    return best_features_dict


