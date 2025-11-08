# -*- coding: utf-8 -*-
# step06-multi_task_train_eval.py
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskElasticNetCV, LinearRegression , Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import LeaveOneOut
import mlflow
from joblib import Parallel, delayed


# ---- helper function for one model/combo ----
def evaluate_model_combo(name, model, X_train, X_test, y_train, y_test, combo_safe, safe_to_orig):
    """Train + evaluate one (model, feature combo) pair"""
    X_train_sel = X_train[list(combo_safe)]
    X_test_sel = X_test[list(combo_safe)]

    model.fit(X_train_sel, y_train)
    y_train_pred = model.predict(X_train_sel)
    y_test_pred = model.predict(X_test_sel)

    R2_train = r2_score(y_train, y_train_pred, multioutput='uniform_average')
    R2_test = r2_score(y_test, y_test_pred, multioutput='uniform_average')
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    return pd.DataFrame({
        'model_name': [name],
        'num_features': [len(combo_safe)],
        'descriptors': [[safe_to_orig[c] for c in combo_safe]],
        'R2_train': [R2_train],
        'R2_test': [R2_test],
        'mae_train': [mae_train],
        'mae_test': [mae_test],
        'mse_train': [mse_train],
        'mse_test': [mse_test],
        'rmse_train': [rmse_train],
        'rmse_test': [rmse_test],
    })

def clone_model(model):
    """Safely clone MultiOutputRegressor or single-output models."""
    if hasattr(model, 'estimator'):
        base = model.estimator.__class__(**model.estimator.get_params())
        return model.__class__(base)
    else:
        return model.__class__(**model.get_params())
    
# ---- main function ----
def run_multitask_models(X_train, X_test, y_train, y_test, selected_features, n_jobs=-1):
    results_df = pd.DataFrame()

    # Safe column renaming for mlflow
    orig_cols = X_train.columns.tolist()
    safe_cols = [c.replace("[","\\").replace("]","\\")
                    .replace("(","//").replace(")","//")
                for c in orig_cols]
    safe_to_orig = dict(zip(safe_cols, orig_cols))

    X_train_safe = X_train.copy()
    X_test_safe = X_test.copy()
    X_train_safe.columns = safe_cols
    X_test_safe.columns = safe_cols

    safe_selected_features = [c.replace("[","\\").replace("]","\\")
                                  .replace("(","//").replace(")","//") 
                              for c in selected_features]

        # Define multitask models
    models = {  
        "GB": MultiOutputRegressor(GradientBoostingRegressor()),
   
    }


    print("Columns in X_train:", X_train.columns.tolist())
    print("Columns in combo:", selected_features)

    mlflow.set_experiment("Neo-tanshinlactone_MultitaskLR")

    with mlflow.start_run():
        tasks = []
        for j in range(1, 11):
            for combo_safe in combinations(safe_selected_features, j):
                for name, model in models.items():
                    tasks.append((name, model, combo_safe))

        # Parallel evaluation across tasks
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(evaluate_model_combo)(
                name,
                clone_model(model),
                X_train_safe,
                X_test_safe,
                y_train,
                y_test,
                combo_safe,
                safe_to_orig
            )

            for name, model, combo_safe in tasks
        )

        results_df = pd.concat(results, ignore_index=True)

    # Sort and find best
    results_df = results_df.sort_values(by='R2_test', ascending=False).reset_index(drop=True)
    results_df.to_csv(r'outputs/results_df.csv', index=False)
    top_model_row = results_df.iloc[0]
    top_model_name = top_model_row['model_name']
    top_features = list(top_model_row['descriptors'])
    print(f" Selected best model: {top_model_name} with features {top_features}")

    # Retrain best model on full training set
    best_model = models[top_model_name]
    best_model.fit(X_train[top_features], y_train)

    return results_df, top_model_row, best_model, top_model_name, top_features
