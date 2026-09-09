import os
import numpy as np
import pandas as pd
import warnings
from sklearn.datasets import load_diabetes, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# 1. Load real datasets


def load_liver_from_file(path="bupa.data"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Download the UCI Liver Disorders "
            f"data file (bupa.data) and place it next to this script, or "
            f"pass the correct path to load_liver_from_file()."
        )
    data = np.loadtxt(path, delimiter=",")
    X = data[:, :5]  # mcv, alkphos, sgpt, sgot, gammagt (drop drinks, selector)
    return X


def load_real_datasets(liver_path="bupa.data"):
    datasets = {}

    # Diabetes (bundled with scikit-learn, no network needed)
    datasets["Diabetes"] = load_diabetes().data

    # Wine (bundled with scikit-learn, no network needed)
    datasets["Wine"] = load_wine().data

    # Liver Disorders (loaded from a local file, see load_liver_from_file above)
    datasets["Liver"] = load_liver_from_file(liver_path)

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
