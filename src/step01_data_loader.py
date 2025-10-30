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
    df = pd.read_csv(file_path)
    required_columns = ['Name', 'SMILES', 'MCF-7', 'SK-BR-3']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    return df
