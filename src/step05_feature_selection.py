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

def feature_selection_results(X, y,  n_features=20, n_estimators=200, random_state=42):
    def run_mtlasso_rfe_per_target(X, y, n_features=20):
        """
        Perform joint (multi-task) feature selection using MultiTaskLasso with RFE.
        Returns a dictionary: {n_features: list of top features}
        """

        best_features_dict = {}

        # Ensure Y is 2D (multi-task: each column is one target)
        if isinstance(y, pd.Series):
            y = y.to_frame()
        elif isinstance(y, np.ndarray) and Y.ndim == 1:
            y = y.reshape(-1, 1)

        print("X info:")
        print(X.info())
        print("y info:")
        print(y.info())
        print("X head:")
        print(X.head())
        print("y head:")
        print(y.head())

         # Initialize MultiTaskLasso estimator
        estimator = MultiTaskLasso(alpha=0.01, random_state=42)

        # RFE wrapper
        selector = RFE(estimator, n_features_to_select=n_features)

        # Fit RFE
        selector.fit(X, y)

        # Get top features
        top_features_list = X.columns[selector.support_].tolist()

        # Return as DataFrame
        final_features_df = pd.DataFrame(top_features_list, columns=["selected_features"])
        print(f"Selected top {n_features} features with MultiTaskLasso + RFE:", top_features_list)

        return top_features_list, final_features_df




    def multitask_xgb_feature_selection(X, y, n_estimators=200, random_state=42, n_features=20):
        """
        Perform multitask feature selection using XGBoost and MultiOutputRegressor.

        """
        # Ensure y is 2D
        if isinstance(y, pd.Series):
            y = y.to_frame()
        elif isinstance(y, np.ndarray) and y.ndim == 1:
            y = y.reshape(-1, 1)

        print("X info:")
        print(X.info())
        print("y info:")
        print(y.info())
        print("X head:")
        print(X.head())
        print("y head:")
        print(y.head())


        orig_cols = X.columns.tolist()
        safe_cols = [c.replace("[","\\").replace("]","\\")
                    .replace("(","//").replace(")","//")
                for c in orig_cols]
        
         # Recover original feature names
        safe_to_orig = dict(zip(safe_cols, orig_cols))
           
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

        feature_ranking['feature'] = feature_ranking['feature_safe'].map(safe_to_orig)

        # Sort by importance
        feature_ranking = feature_ranking.sort_values('importance', ascending=False)
        print(feature_ranking.head(n_features))
        
        feature_list = feature_ranking['feature'].tolist() [:n_features] # get the list of feature names
        final_features_df = pd.DataFrame(feature_list, columns=["selected_features"])

        print(f"Selected top {n_features} features using XGB feature importance:")

        return  feature_list, final_features_df

    top_features_list_rfe, _ = run_mtlasso_rfe_per_target  (X, y, n_features=15)
    
    # Run multi-task XGB feature selection
    final_features_list_xgb, _ =  multitask_xgb_feature_selection(X, y, n_estimators=200, random_state=42, n_features=15)

    for item in final_features_list_xgb:
        if item not in top_features_list_rfe:
            top_features_list_rfe.append(item)
    top_list = top_features_list_rfe

    top_features_df = pd.DataFrame(top_list, columns=["selected_features"])
    print(f"Selected top {n_features} features:", top_features_df)

    return top_list , top_features_df
