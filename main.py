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
    normalize_features,
    run_mtlasso_rfe_per_target,   
    multitask_xgb_feature_selection,
    run_multitask_models,   # multitask LR training/evaluation
    visualize_model,
    save_model, 
    cluster_based_train_test_split, 
   split_train_test
)

# Configuration
DATA_PATH = "data/Neo-tanshinlactone_cytotoxicity.csv"   # Input CSV
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "multi_task_lr_model.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "selected_features.csv")


# 1 Load Data
print(" Loading cytotoxicity data...")
df, X, y, target_columns = load_data(DATA_PATH)
print(f" Loaded {len(df)} compounds with columns: {df.columns.tolist()}")


# 2 Generate RDKit Descriptors if ising RDkit to calculate the descriptors from MOL files otherwise comment if you use your own descriptors

print(" Generating RDKit molecular descriptors (all available)...")
# df_desc = smiles_to_all_descriptors(df, corr_threshold=0.95) 
# print(f" Generated {df_desc.shape[1] - 4} molecular descriptors.")

df_desc = df = pd.read_csv( r"data/descriptors.csv", encoding="iso-8859-1")


print(f"  {df_desc.shape[1] - 4} molecular descriptors.")
print(df_desc.head())

#  Prepare Feature Matrix (X) and Targets (y)
# Adjust column names to your dataset: 'Compound' and 'SMILE'
descriptor_cols = [c for c in df_desc.columns if c not in ['Compound', 'SMILE' ] + target_columns]

X = df_desc[descriptor_cols]
y = df_desc[target_columns]
print ('X', X)
print ('y', y)


# 3. Split before feature selection (to avoid leakage)
print(" Splitting data into train/test sets...")

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

X_train, X_test, y_train, y_test = cluster_based_train_test_split(
        data=df_desc,
        descriptor_cols=descriptor_cols,
        target_cols=target_columns,
        test_size=0.2,
        random_state=42
    )

print ('X_train', X_train)
print ('y_train', y_train)

print ('X_test', X_test)
print ('y_test', y_test)
# 4: Preprocessing 
print("Step 4: Preprocessing data...")
X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)


# 5. Feature Selection using MultiTaskLasso + RFE or  MultiTask xgb
# print(" Running MultiTaskLasso + RFE for feature selection...")
# final_features_df, final_features = run_mtlasso_rfe_per_target(X_train_scaled, y_train , n_features_list=[1, 2, 3, 4])

print ('X_train.column',X_train.columns )  

print(" Running MultiTask xgb for feature selection...")
final_features_df, final_features  = multitask_xgb_feature_selection(X_train, y_train, top_n=20)

# Save selected features
final_features_df.to_csv(FEATURES_PATH)
print("final features", final_features)

# # Use the largest subset (best 4 vars) for final multitask model
# selected_features = best_features[max(best_features.keys())]
# print ('selected_features' , selected_features)


# Apply same selected features to both train/test sets
X_train_selected = X_train_scaled[final_features]
X_test_selected = X_test_scaled[final_features]


# 6 Train Multitask Linear Regression (MCF-7 & SK-BR-3)
print(" Training multitask linear regression model...")
results_df, top_model_row, best_model, top_model_name = run_multitask_models(X_train, X_test, y_train, y_test, final_features)
print(" Training complete. Evaluation metrics:")
print(results_df)


# Step 7: Model visualiation & Plots  
print("Step 6: Evaluating models and generating plots...")

SAVE_DIR = "outputs/plots"
descriptor_cols = list(top_model_row.iloc[2])

visualize_model(
X_train=X_train_selected[descriptor_cols],
X_test=X_test_selected[descriptor_cols],
y_train=y_train,
y_train_pred=best_model.predict(X_train_selected[descriptor_cols]),
y_test=y_test,
y_test_pred=best_model.predict(X_test_selected[descriptor_cols]),
target_names = target_columns,
best_model_name=top_model_name,
save_dir = SAVE_DIR,
)
print("Pipeline completed successfully.")

# 8 Save model and metrics
print(" Saving model and metrics...")
save_model(best_model , MODEL_PATH)
results_df.to_csv(METRICS_PATH, index=False)

print("\n Pipeline finished successfully!")
print(f" Model saved to: {MODEL_PATH}")
print(f" Metrics saved to: {METRICS_PATH}")
print(f" Selected features saved to: {FEATURES_PATH}")
