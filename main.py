# -*- coding: utf-8 -*-
"""
Main MLOps pipeline for Neo-tanshinlactone compounds:
- Load data
- Generate RDKit descriptors
- Select features using Genetic Algorithm (GA)
- Train and evaluate multitask Linear Regression
- Save trained model and results

Author: RASULEVLAB
Date: Oct 28, 2025
"""

# --- Step Imports ---
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src import (
    load_data,
    smiles_to_all_descriptors,
    run_mtlasso_rfe_per_target,   
    run_multitask_models,   # multitask LR training/evaluation
    save_model
)

# Configuration
DATA_PATH = "data/Neo-tanshinlactone_cytotoxicity.csv"   # Input CSV
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "multi_task_lr_model.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "selected_features.csv")


# 1 Load Data
print(" Loading cytotoxicity data...")
df = load_data(DATA_PATH)
print(f" Loaded {len(df)} compounds with columns: {df.columns.tolist()}")


# 2 Generate RDKit Descriptors
print(" Generating RDKit molecular descriptors (all available)...")
df_desc = smiles_to_all_descriptors(df, corr_threshold=0.95) 
print(f" Generated {df_desc.shape[1] - 4} molecular descriptors.")
print(df_desc.head())

# 3 Prepare Feature Matrix (X) and Targets (y)
X = df_desc.drop(columns=["Name", "SMILES", "MCF-7", "SK-BR-3"])
y = df_desc[["MCF-7", "SK-BR-3"]]


# 4. Feature Selection using MultiTaskLasso + RFE
print(" Running MultiTaskLasso + RFE for feature selection...")
best_features = run_mtlasso_rfe_per_target(X, y, n_features_list=[1, 2, 3, 4])
print(" Best features per subset size (based on multitask target):")
for n, feats in best_features.items():
    print(f"   • {n} features: {feats}")

# Save selected features
pd.DataFrame.from_dict(best_features, orient='index').to_csv(FEATURES_PATH)

# Use the largest subset (best 4 vars) for final multitask model
selected_features = best_features[max(best_features.keys())]
X_selected = X[selected_features]

#  Split data into train/test
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# 6 Train Multitask Linear Regression (MCF-7 & SK-BR-3)
print(" Training multitask linear regression model...")
results_df, best_model = run_multitask_models(X_train, X_test, y_train, y_test, selected_features)
print(" Training complete. Evaluation metrics:")
print(results_df)


# 7 Save model and metrics
print(" Saving model and metrics...")
save_model(best_model , MODEL_PATH)
results_df.to_csv(METRICS_PATH, index=False)

print("\n Pipeline finished successfully!")
print(f" Model saved to: {MODEL_PATH}")
print(f" Metrics saved to: {METRICS_PATH}")
print(f" Selected features saved to: {FEATURES_PATH}")
