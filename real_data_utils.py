import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from highdim_gene_data import load_colon_cancer, load_leukemia_combined, load_riboflavin
from libsvm_utils import variance_screen


# 1. Load real datasets


def load_real_datasets(p_screen=300):
    datasets = {}

    Xc = load_colon_cancer()
    datasets["Colon"], _ = variance_screen(Xc, p_screen)

    Xl = load_leukemia_combined()
    datasets["Leukemia"], _ = variance_screen(Xl, p_screen)

    Xr, _ = load_riboflavin()
    datasets["Riboflavin"], _ = variance_screen(Xr, p_screen)

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
