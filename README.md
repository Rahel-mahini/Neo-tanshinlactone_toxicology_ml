# Neo-Tanshinlactone Cytotoxicity Prediction Pipeline

This repository contains a **multi-task machine learning pipeline** to predict cytotoxicity of Neo-Tanshinlactone compounds across multiple cell lines using **molecular descriptors** generated from SMILES strings.

---

## 🔹 Project Overview

The pipeline performs the following steps:

1. **Load data**  
   Reads a CSV file containing compound information, SMILES strings, and cytotoxicity targets (MCF-7, SK-BR-3).

2. **Generate molecular descriptors**  
   Uses **RDKit** to compute all available molecular descriptors. Constant, NaN, or highly correlated descriptors are automatically removed.

3. **Feature selection**  
   - Uses **MultiTaskLasso + Recursive Feature Elimination (RFE)** to select top features for all targets simultaneously.  
   - Provides top 1–4 features for downstream modeling.

4. **Train multi-task regression model**  
   - Fits **multi-task linear regression** using the selected descriptors.  
   - Evaluates models using R², MAE, MSE, RMSE, and leave-one-out R² metrics.  
   - Supports training with 1, 2, 3, up to 4 features combinations.

5. **Save results**  
   - Trained model saved as `.pkl`.  
   - Models evaluation metrics saved as `.csv`.  
   - Selected features saved as `.csv`.

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/Rahel-mahini/Neo-tanshinlactone_toxicology_ml.git
cd neo_tanshinlactone_ml
pip install -r requirements.txt
python main.py

```