# step09_utils_metrics.py

import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def loo_r2_score(model, X, y):
    """
    Compute Leave-One-Out R² score
    """
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    return r2_score(y, preds)

def compute_metrics(model, X_train, X_test, y_train, y_test, target_name="Target"):
    """
    Compute LOO R², R² train/test, MAE train/test, RMSE train/test
    """
    # Train model
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    metrics = {
        "Target": target_name,
        "LOO_R2": loo_r2_score(model, X_train, y_train),
        "R2_train": r2_score(y_train, y_pred_train),
        "R2_test": r2_score(y_test, y_pred_test),
        "MAE_train": mean_absolute_error(y_train, y_pred_train),
        "MAE_test": mean_absolute_error(y_test, y_pred_test),
        "RMSE_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_test))
    }
    return metrics
