# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:49:10 2025

@author: RASULEVLAB
"""

# step02-feature_engineering.py
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

def smiles_to_all_descriptors(df: pd.DataFrame, corr_threshold: float = 0.95) -> pd.DataFrame:
    """
    Convert SMILES strings to ALL available RDKit molecular descriptors.
    Automatically retrieves all descriptor functions from rdkit.Chem.Descriptors.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a column 'SMILES' containing molecular SMILES strings.

    Returns
    -------
    pd.DataFrame
        Original dataframe with appended RDKit descriptor columns.
    """
    # Get all RDKit descriptor functions dynamically
    descriptor_funcs = [(name, func) for name, func in Descriptors.descList]
    descriptor_names = [name for name, _ in descriptor_funcs]

    def calc_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Return NaN if SMILES invalid
            return [float('nan')] * len(descriptor_funcs)
        values = []
        for _, func in descriptor_funcs:
            try:
                values.append(func(mol))
            except Exception:
                values.append(float('nan'))
        return values

    # Compute all descriptors
    descriptors = df['SMILES'].apply(calc_descriptors).tolist()
    descriptor_df = pd.DataFrame(descriptors, columns=descriptor_names)

    # Drop descriptors that are all NaN
    descriptor_df = descriptor_df.dropna(axis=1, how='all')

    # Drop descriptors with constant values
    descriptor_df = descriptor_df.loc[:, descriptor_df.nunique() > 1]

    # Drop highly correlated descriptors
    corr_matrix = descriptor_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    descriptor_df = descriptor_df.drop(columns=to_drop)

    # Merge descriptors with original dataframe
    result_df = pd.concat([df.reset_index(drop=True), descriptor_df], axis=1)
    return result_df
