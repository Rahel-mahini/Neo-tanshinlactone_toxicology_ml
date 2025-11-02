# -*- coding: utf-8 -*-
"""
step01_data_loader.py
Load raw descriptors and target data.
"""

import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load cytotoxicity data CSV with columns: Name, SMILES, MCF-7, SK-BR-3
    """
    df = pd.read_csv(file_path, encoding="iso-8859-1")

    # Strip whitespace and non-breaking characters from column names
    df.columns = [col.strip().replace('\xa0', '') for col in df.columns]
    print("Columns cleaned:", df.columns)
    
    # Ensure columns exist
    if 'Compound' not in df.columns or 'SMILE' not in df.columns:
        raise ValueError("Dataset must contain 'Name' and 'SMILES' columns.")

    # Detect potential target columns (numeric types)
    potential_targets = df.select_dtypes(include=['float', 'int']).columns.tolist()

    # Remove duplicates just in case
    potential_targets = [col for col in potential_targets if col not in ['Compound', 'SMILE']]

    # Identify targets automatically
    target_columns = potential_targets
    print(f" Detected target columns: {target_columns}")

    # Prepare X (features) and y (targets)
    X = df.drop(columns=['Compound', 'SMILE'] + target_columns)
    y = df[target_columns]

    return df, X, y, target_columns