import numpy as np

import pandas as pd

import warnings

from sklearn.datasets import load_diabetes, load_wine, fetch_openml

from sklearn.preprocessing import StandardScaler

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# 1. Load four real datasets

def load_real_datasets():

    datasets = {}

    # Diabetes

    datasets["Diabetes"] = load_diabetes().data

    # Wine

    datasets["Wine"] = load_wine().data

    # Liver Disorders 

    liver = fetch_openml(name='liver-disorders', as_frame=True)

    datasets["Liver"] = liver.data.values  

    return datasets



# 2. Preprocess real X

def preprocess_X(X):
    X = np.asarray(X, dtype=float)

    # Remove columns with almost zero variance
    col_sd = X.std(axis=0)
    X = X[:, col_sd > 1e-8]

    # Standardize columns
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X



# 3. Generate semi-synthetic response from real X

def generate_semisynthetic_from_real_X(
    X,
    s=10,
    sigma=1.0,
    beta_low=1.0,
    beta_high=2.0,
    seed=1
):
    np.random.seed(seed)

    n, p = X.shape
    s_use = min(s, max(1, p // 3))

    beta = np.zeros(p)
    support = np.random.choice(p, s_use, replace=False)
    beta[support] = (
        np.random.choice([-1, 1], size=s_use)
        * np.random.uniform(beta_low, beta_high, size=s_use)
    )

    y = X @ beta + np.random.normal(0, sigma, size=n)

    return y, beta
