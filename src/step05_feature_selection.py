# -*- coding: utf-8 -*-
"""
Feature selection using Gentic Algorithm
"""
# step05_feature_selection.py
from sklearn.linear_model import MultiTaskLasso
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


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
    
     # Create a DataFrame from the results
    best_features_df = pd.DataFrame.from_dict(best_features_dict, orient='index')

    # Use the largest subset (e.g. max number of features)
    final_features = best_features_dict[max(best_features_dict.keys())]
    final_features_df = pd.DataFrame(final_features, columns=["selected_features"])
    print('Selected features (final):', final_features)

    return best_features_df, final_features_df




def multitask_xgb_feature_selection(X, y, n_estimators=200, random_state=42, top_n=None):
    """
    Perform multitask feature selection using XGBoost and MultiOutputRegressor.

    Parameters
    ----------
    X : pd.DataFrame
        Feature/descriptor matrix.
    y : pd.DataFrame
        Multi-target values.
    n_estimators : int, default=200
        Number of boosting rounds for XGBRegressor.
    random_state : int, default=42
        Random seed for reproducibility.
    top_n : int or None, default=None
        Number of top features to return. If None, returns all features.

    Returns
    -------
    feature_ranking : pd.DataFrame
        DataFrame with columns ['feature', 'importance'] sorted by descending importance.
    """
    orig_cols = X.columns.tolist()
    safe_cols = [c.replace("[","\\").replace("]","\\")
                .replace("(","//").replace(")","//")
             for c in orig_cols]
        
    # Create copies of your train/test sets with safe column names
    X_safe = X.copy()
    X_safe.columns = safe_cols


    # Wrap XGBoost for multitask regression
    model = MultiOutputRegressor(
        XGBRegressor(tree_method='hist', n_estimators=n_estimators, random_state=random_state)
    )
    
    # Fit model
    model.fit(X_safe, y)
    
    # Extract feature importance per target
    importances_per_target = [est.feature_importances_ for est in model.estimators_]
    
    # Average feature importance across all targets
    average_importance = np.mean(importances_per_target, axis=0)
    
    # Rank features
    feature_ranking = pd.DataFrame({
        'feature_safe': X_safe.columns,
        'importance': average_importance
    })  

    # Recover original feature names
    safe_to_orig = dict(zip(safe_cols, orig_cols))
    feature_ranking['feature'] = feature_ranking['feature_safe'].map(safe_to_orig)

    # Sort by importance
    feature_ranking = feature_ranking.sort_values('importance', ascending=False)
    print(feature_ranking.head(top_n))
    
    feature_list = feature_ranking['feature'].tolist()  # get the list of feature names
    final_features_df = pd.DataFrame(feature_list, columns=["selected_features"])

    return  final_features_df, feature_list
