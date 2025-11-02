# -*- coding: utf-8 -*-

# step08_model_serving.py

import pickle

def save_model(model, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)
