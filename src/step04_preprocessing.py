# -*- coding: utf-8 -*-
# step04_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled

def handle_missing(X: pd.DataFrame):
    return X.fillna(X.mean())