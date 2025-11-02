# -*- coding: utf-8 -*-
# step07_model_visualization.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from numpy.linalg import pinv
import os
import warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, MultiTaskLassoCV, MultiTaskElasticNetCV, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from alibi.explainers import ALE

def visualize_model(X_train, X_test, y_train, y_train_pred, y_test, y_test_pred, target_names, best_model_name, save_dir ):
    """
    Generate correlation, Williams, and ALE plots for the trained model.

    Args:
        X_train, X_test: pd.DataFrame
        y_train, y_test: pd.DataFrame or np.array
        y_train_pred, y_test_pred: np.array
        target_names: list of target names
        best_model_name: str -> name of model ('MultiTaskLinear', 'MultiTaskLasso', or 'MultiTaskElasticNet')
        save_dir: str -> directory to save plots
    """
    def correlation_plot_multitask():
        """
        Save separate correlation plots for each target in multitask regression.
        """
        os.makedirs(save_dir, exist_ok=True)
        n_targets = y_train.shape[1]

        y_train_np = np.array(y_train)
        y_train_pred_np = np.array(y_train_pred)

        for i in range(n_targets):
            plt.figure(figsize=(6, 5))
            plt.scatter(y_train_np[:, i], y_train_pred_np[:, i],
                        alpha=0.7, s=100, color='skyblue')
            plt.plot(
                [y_train_np[:, i].min(), y_train_np[:, i].max()],
                [y_train_np[:, i].min(), y_train_np[:, i].max()],
                color='k', linestyle='--'
            )
            r2 = r2_score(y_train_np[:, i], y_train_pred_np[:, i])
            plt.title(f'{target_names[i]} (R²={r2:.3f})')
            plt.xlabel("Experimental")
            plt.ylabel("Predicted")

            file_path = os.path.join(save_dir, f"correlation_{target_names[i]}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved correlation plot for {target_names[i]}: {file_path}")


    def williams_plot_multitask():
        """
        Generate and save Williams plots per target for multitask regression.
        
        Args:
            X_train, X_test: pd.DataFrame or np.array of features
            y_train, y_test: pd.DataFrame or np.array of true target values
            y_train_pred, y_test_pred: np.array of predicted values
            target_names: list of target names
            save_dir: folder to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        X_train_np = np.array(X_train)
        X_test_np = np.array(X_test)
        y_train_np = np.array(y_train)
        y_test_np = np.array(y_test)
        
        n_train, p = X_train_np.shape
        threshold_leverage = 3*(p+1)/n_train
        threshold_resid = 3

        # Hat matrices
        H_train = X_train_np @ pinv(X_train_np.T @ X_train_np) @ X_train_np.T
        leverage_train = np.diag(H_train)
        
        H_test = X_test_np @ pinv(X_train_np.T @ X_train_np) @ X_test_np.T
        leverage_test = np.diag(H_test)

        for i, target in enumerate(target_names):
            std_res_train = (y_train_np[:, i] - y_train_pred[:, i]) / np.std(y_train_np[:, i] - y_train_pred[:, i])
            std_res_test = (y_test_np[:, i] - y_test_pred[:, i]) / np.std(y_test_np[:, i] - y_test_pred[:, i])

            plt.figure(figsize=(8, 8))
            plt.scatter(leverage_train, std_res_train, alpha=0.6, s=100, label='Training', color='skyblue')
            plt.scatter(leverage_test, std_res_test, alpha=0.6, s=100, label='Test', color='#007FDE')
            plt.axhline(y=threshold_resid, color='k', linestyle='--')
            plt.axhline(y=-threshold_resid, color='k', linestyle='--')
            plt.axvline(x=threshold_leverage, color='k', linestyle='--')
            plt.xlabel('Leverage')
            plt.ylabel('Standardized Residuals')
            plt.title(f'Williams Plot - {target}')
            plt.legend()
        
            file_path = os.path.join(save_dir, f"williams_{target}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
            print(f"Williams plot saved for {target}: {file_path}")


    def evaluate_model_ale_multitask():
        """
        Fit a multitask model and generate ALE plots for each target.
        
        Args:
            X_train: pd.DataFrame of features
            y_train: pd.DataFrame or np.array of shape (n_samples, n_targets)
            best_model_name: str, one of ["MultiTaskLinear", "MultiTaskLasso", "MultiTaskElasticNet"]
            save_dir: directory to save ALE plots
        """
        warnings.filterwarnings("ignore")
        
        # Normalize features
        scaler = MinMaxScaler()
        X_train_normalized = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        
        # Initialize model
   
        models = {
            "MultiTaskLinear": LinearRegression(),
            "MultiTaskLasso": MultiTaskLassoCV(cv=5),
            "MultiTaskElasticNet": MultiTaskElasticNetCV(cv=5), 
            'Ridge' :MultiOutputRegressor(Ridge(random_state=42)),
            'RandomForestRegressor' : MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
            "GB": MultiOutputRegressor(GradientBoostingRegressor()),
            "SVR": MultiOutputRegressor(SVR()),
            "MLP": MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(50, 20), max_iter=500)),
            'XGB':  MultiOutputRegressor(XGBRegressor(tree_method='hist', n_estimators=200)),
            "DecisionTree": MultiOutputRegressor(DecisionTreeRegressor(random_state=42)),
            "GaussianProcess": MultiOutputRegressor(GaussianProcessRegressor())
        }
        
        if best_model_name not in models:
            raise ValueError(f"Unknown model: {best_model_name}")
        
        model = models[best_model_name]
        
        # Fit the model
        model.fit(X_train_normalized, y_train)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get target names
        if hasattr(y_train, "columns"):
            target_names = y_train.columns.tolist()
        else:
            target_names = [f"Target_{i}" for i in range(y_train.shape[1])]
        
        # Compute and plot ALE per target
        for t_idx, target in enumerate(target_names):
            def predict_target(X):
                pred = model.predict(X)
                # select the target column
                return pred[:, t_idx].reshape(-1, 1)
            
            feature_names = list(X_train_normalized.columns)
            ale = ALE(predict_target, feature_names=feature_names, target_names=[target])
            ale_exp = ale.explain(X_train_normalized.values)
            
            # Plot ALE for each feature
            for i, feature_name in enumerate(ale_exp.feature_names):
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.plot(ale_exp.feature_values[i], ale_exp.ale_values[i])
                ax.set_title(f'ALE Plot - {feature_name} ({target})', fontsize=16)
                ax.set_xlabel(feature_name, fontsize=14)
                ax.set_ylabel('ALE Value', fontsize=14)
                plt.tick_params(axis='x', labelsize=12)
                plt.tick_params(axis='y', labelsize=12)
                ale_path = os.path.join(save_dir, f"ALE_plot_{target}_{feature_name}.png")
                plt.savefig(ale_path, dpi=600, transparent=True)
                plt.close()
                print(f"ALE plot saved at: {ale_path}")

    

    # Call correlation plot
    correlation_plot_multitask()

    # Call Williams plot
    williams_plot_multitask()

    # Call ALE plot
    ale_exp = evaluate_model_ale_multitask()
    print("All plots saved successfully.")

    return ale_exp