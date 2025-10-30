# -*- coding: utf-8 -*-
# step08_visualization.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from numpy.linalg import pinv
import os
import warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from alibi.explainers import ALE, plot_ale

def visualize_model(X_train, X_test, y_train, y_train_pred, y_test, y_test_pred, save_dir,  config):
    def correlation_plot(y_train, y_train_pred, y_test, y_test_pred, save_dir):
        
        os.makedirs(save_dir, exist_ok=True)
        corr_path = os.path.join(save_dir, "correlation_plot.png")
        plt.figure(figsize=(8,6))
        plt.scatter(y_train, y_train_pred, color='skyblue', label='Training', alpha=0.7, s=100)
        plt.scatter(y_test, y_test_pred, color='#007FDE', label='Test', alpha=0.7, s=100)
        plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='k', linestyle='--', linewidth=2)
        plt.xlabel('Experimental')
        plt.ylabel('Predicted')
        plt.legend()
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        plt.text(0.95, 0.05, f'R2 Train: {r2_train:.3f}\nR2 Test: {r2_test:.3f}', 
                horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)
        plt.savefig(corr_path, dpi=600, bbox_inches='tight', transparent=True)

    def williams_plot(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred, save_dir):

        X_train_np = np.array(X_train)
        X_test_np = np.array(X_test)
        y_train_np = np.array(y_train)
        y_test_np = np.array(y_test)

        williams_path = os.path.join(save_dir, "williams_plot.png")
    
        # Hat matrices
        H_train = X_train_np @ pinv(X_train_np.T @ X_train_np) @ X_train_np.T
        leverage_train = np.diag(H_train)
        
        H_test = X_test_np @ pinv(X_train_np.T @ X_train_np) @ X_test_np.T
        leverage_test = np.diag(H_test)
        
        # Standardized residuals
        std_res_train = (y_train_np - y_train_pred) / np.std(y_train_np - y_train_pred)
        std_res_test = (y_test_np - y_test_pred) / np.std(y_test_np - y_test_pred)
        
        n_train, p = X_train_np.shape
        threshold_leverage = 3*(p+1)/n_train
        threshold_resid = 3
        
        plt.figure(figsize=(8,8))
        plt.scatter(leverage_train, std_res_train, alpha=0.6, s=100, label='Training', color='skyblue')
        plt.scatter(leverage_test, std_res_test, alpha=0.6, s=100, label='Test', color='#007FDE')
        plt.axhline(y=threshold_resid, color='k', linestyle='--')
        plt.axhline(y=-threshold_resid, color='k', linestyle='--')
        plt.axvline(x=threshold_leverage, color='k', linestyle='--')
        plt.xlabel('Leverage')
        plt.ylabel('Standardized Residuals')
        plt.legend()
        plt.savefig(williams_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
        print(f"Williams plot saved at: {williams_path}")


    def evaluate_model_ale(X_train, y_train, save_dir, config):
        """
        Fit a model and generate ALE plots for interpretability.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series or np.array): Training targets
    
            config (dict): YAML configuration containing model parameters and output paths
        """
        warnings.filterwarnings("ignore")
        
        # ---------------- Normalize features ----------------
        scaler = MinMaxScaler()
        X_train_normalized = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

        # ---------------- Initialize model dynamically ----------------
        model_params = config.get('model', {})
        model_type = model_params['type']

        model_mapping = {
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'LinearRegression': LinearRegression,
            'Lasso': Lasso,
            'Ridge': Ridge,
            'RandomForest': RandomForestRegressor,
            'SVR': SVR
        }

        model_kwargs = {k: v for k, v in model_params.items() if k != 'type'}

        # Add random_state if applicable
        if model_type in ['RandomForest', 'DecisionTree'] and 'random_state' not in model_kwargs:
            model_kwargs['random_state'] = 42

        ModelClass = model_mapping[model_type]        
        model = ModelClass(**model_kwargs)

        # ---------------- Fit the model ----------------
        model.fit(X_train_normalized, y_train)

        # ---------------- Compute ALE ----------------
        feature_names = list(X_train_normalized.columns)
        target_name = config.get('plots', {}).get('target_name', 'Target')
        gb_ale = ALE(model.predict, feature_names=feature_names, target_names=[target_name])
        ale_exp = gb_ale.explain(X_train_normalized.values)

        # ---------------- Plot ALE for each feature ----------------
        os.makedirs(save_dir, exist_ok=True)

        for i, feature_name in enumerate(ale_exp.feature_names):
            fig, ax = plt.subplots(figsize=(9, 6))
            # manually plot the ALE values for the i-th feature
            ax.plot(ale_exp.feature_values[i], ale_exp.ale_values[i])
            
            ax.set_title(f'ALE Plot - {feature_name}', fontsize=16)
            ax.set_xlabel(feature_name, fontsize=14)
            ax.set_ylabel('ALE Value', fontsize=14)
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)
            
            ale_path = os.path.join(save_dir, f"ALE_plot_{feature_name}.png")
            plt.savefig(ale_path, dpi=600, transparent=True)
            plt.close()
            print(f"ALE plot saved at: {ale_path}")

 

        # Create save_dir if not exists
    os.makedirs(save_dir, exist_ok=True)

    # Call correlation plot
    correlation_plot(y_train, y_train_pred, y_test, y_test_pred, save_dir)

    # Call Williams plot
    williams_plot(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred, save_dir)

    # Call ALE plot
    ale_exp = evaluate_model_ale(X_train, y_train, save_dir, config)

    print("All plots saved successfully.")

    return ale_exp