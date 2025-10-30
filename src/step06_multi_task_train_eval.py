# -*- coding: utf-8 -*-
# step06-multi_task_train_eval.py
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskElasticNetCV, MultiTaskLinearRegression
from sklearn.model_selection import LeaveOneOut
import mlflow
import mlflow.sklearn

# Function to compute LOO R2
def loo_r2_score(model, X, y):
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred.append(model.predict(X_test)[0])
        y_true.append(y_test.values[0])
    return r2_score(y_true, y_pred)


def run_multitask_models(X_train, X_test, y_train, y_test, selected_features):
    results_df = pd.DataFrame()

    # Define multitask models
    models = {
        "MultiTaskLinear": MultiTaskLinearRegression(),
        "MultiTaskLasso": MultiTaskLassoCV(cv=5),
        "MultiTaskElasticNet": MultiTaskElasticNetCV(cv=5)
    }

    mlflow.set_experiment("Neo-tanshinlactone_MultitaskLR")

    with mlflow.start_run():
        # Loop over 1, 2, 3, and 4-feature combinations
        for j in range(1, len(selected_features) + 1):
            for combo in combinations(selected_features, j):
                X_train_sel = X_train[list(combo)]
                X_test_sel = X_test[list(combo)]

                for name, model in models.items():
                    # Fit and predict
                    model.fit(X_train_sel, y_train)
                    y_train_pred = model.predict(X_train_sel)
                    y_test_pred = model.predict(X_test_sel)

                    # Compute metrics
                    R2_train = r2_score(y_train, y_train_pred, multioutput='uniform_average')
                    R2_test = r2_score(y_test, y_test_pred, multioutput='uniform_average')
                    mae_train = mean_absolute_error(y_train, y_train_pred)
                    mae_test = mean_absolute_error(y_test, y_test_pred)
                    mse_train = mean_squared_error(y_train, y_train_pred)
                    mse_test = mean_squared_error(y_test, y_test_pred)
                    rmse_train = np.sqrt(mse_train)
                    rmse_test = np.sqrt(mse_test)

                    # LOO R2 (optional, can be slow)
                    # loo_r2 = loo_r2_score(model, X_train_sel, y_train)
                    loo_r2 = loo_r2_score(model, X_train_sel, y_train)

                    # Store results
                    results_df = pd.concat([
                        results_df,
                        pd.DataFrame({
                            'model_name': [name],
                            'num_features': [j],
                            'descriptors': [combo],
                            'R2_train': [R2_train],
                            'R2_test': [R2_test],
                            'mae_train': [mae_train],
                            'mae_test': [mae_test],
                            'mse_train': [mse_train],
                            'mse_test': [mse_test],
                            'rmse_train': [rmse_train],
                            'rmse_test': [rmse_test],
                            'loo_r2': [loo_r2]
                        })
                    ], ignore_index=True)

    # Sort by test R² and pick best model
    results_df = results_df.sort_values(by='R2_test', ascending=False).reset_index(drop=True)
    top_model_row = results_df.iloc[0]

    top_model_name = top_model_row['model_name']
    top_features = list(top_model_row['descriptors'])
    print(f"✅ Selected best model: {top_model_name} with features {top_features}")

    # Retrain best model on full training set
    best_model = models[top_model_name]
    best_model.fit(X_train[top_features], y_train)

    return results_df, top_model_row, best_model
