
# -*- coding: utf-8 -*-

# step07_hyperparameter_optimization.py

import pandas as pd
from itertools import product
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import Parallel, delayed
import math
import xgboost as xgb
from xgboost import XGBRegressor
import time
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

# --- helper worker that trains one model and returns metrics ---
def _evaluate_param_combination(param_dict,
                                X_train, X_test,
                                y_train, y_test,
                                random_state=42):
    """
    Train one XGBRegressor with given param_dict and return metrics as dict.
    Note: inside XGB we set n_jobs=1 to avoid nested parallelism.
    """
    # ensure model uses single-thread internally (avoid nested threading)
    safe_params = dict(param_dict)
    safe_params.setdefault("random_state", random_state)
    # prefer tree_method 'hist' for speed on larger datasets; leave to param_grid otherwise
    #safe_params.setdefault("tree_method", "hist")
    #safe_params["n_jobs"] = 1

    model = MultiOutputRegressor(GradientBoostingRegressor(**safe_params))

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # compute metrics
    R2_train = r2_score(y_train, y_train_pred)
    R2_test = r2_score(y_test, y_test_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_train = math.sqrt(mse_train)
    rmse_test = math.sqrt(mse_test)

    result = dict(safe_params) 

    # pack result (coerce param values to simple serializable types)
    result.update ({
        "model_name": "GradientBoostingRegressor",
        "R2_train": R2_train,
        "R2_test": R2_test,
        "mae_train": mae_train,
        "mae_test": mae_test,
        "mse_train": mse_train,
        "mse_test": mse_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "train_time_s": train_time
    })

    return result

# --- run parallel grid (replace your for params in param_combinations loop) ---
def run_param_search_parallel(X_train, X_test,
                              y_train, y_test,
                              param_grid_gb,
                              n_jobs=-1,  # joblib parallel workers
                              random_state=42,
                              results_path=None,
                              results_name="combination_para_opt_gb.csv"):
    """
    Evaluate all parameter combinations in parallel using joblib.
    - param_grid_gb: dict of lists (same shape as you defined)
    - n_jobs: number of parallel workers (use -1 for all cores)
    """
    # build list of param dicts
    keys = list(param_grid_gb.keys())
    vals = list(param_grid_gb.values())
    param_combinations = [dict(zip(keys, combo)) for combo in product(*vals)]

    # defensive: convert selected X to numpy arrays for faster serialization between processes
    # (joblib will pickle/transfer them anyway)
    X_train_sel = X_train.copy()
    X_test_sel = X_test.copy()

    # run jobs in parallel
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_evaluate_param_combination)(
            param_dict,
            X_train_sel, X_test_sel,
            y_train, y_test,
            random_state
        ) for param_dict in param_combinations
    )

    results_df = pd.DataFrame(results)

    # add descriptors / column list info if desired (same for all runs)
    results_df["descriptors"] = [", ".join(X_train.columns)] * len(results_df)

    # sort and save if path provided
    results_df = results_df.sort_values(by="R2_test", ascending=False).reset_index(drop=True)

    if results_path:
        os.makedirs(results_path, exist_ok=True)
        results_df.to_csv(os.path.join(results_path, results_name), index=False)

    
    # Train final best model
    best_params = results_df.iloc[0][param_grid_gb.keys()].to_dict()
    best_model = MultiOutputRegressor(GradientBoostingRegressor(random_state=random_state, **best_params)) 
    best_model.fit(X_train, y_train)
    top_model_row = results_df.iloc[0]

    return results_df, top_model_row, best_model



# def detect_gpu_support():
#     """Return True if XGBoost can run with GPU support."""
#     try:
#         # Try to initialize a small GPU model
#         test_model = XGBRegressor(tree_method="gpu_hist") 
#         return True
#     except xgb.core.XGBoostError as e:
#         if "GPU" in str(e):
#             return False
#         raise  # if it’s another kind of error, show it