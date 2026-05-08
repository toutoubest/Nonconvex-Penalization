import numpy as np
from numpy.linalg import eigvalsh
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler



# 1. Generate toy data

def generate_data(n=200, p=100, s=10, rho=0.5, sigma=1.0, seed=1):
    np.random.seed(seed)

    Sigma = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)

    beta = np.zeros(p)
    support = np.random.choice(p, s, replace=False)
    beta[support] = np.random.choice([-1, 1], size=s) * np.random.uniform(1.0, 2.0, size=s)

    y = X @ beta + np.random.normal(0, sigma, size=n)

    return X, y, beta, Sigma
