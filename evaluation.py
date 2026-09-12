import numpy as np
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from optimization import fit_quadratic_penalty


# Optional post-selection thresholding:

def post_threshold(beta, threshold=1e-3):
    beta_new = beta.copy()
    beta_new[np.abs(beta_new) < threshold] = 0.0
    return beta_new


# Evaluation metrics:


def evaluate(beta_hat, beta_true, threshold=1e-3):
    selected = np.abs(beta_hat) > threshold
    true_support = beta_true != 0

    mse = np.mean((beta_hat - beta_true) ** 2)

    try:
        auc = roc_auc_score(true_support.astype(int), np.abs(beta_hat))
    except Exception:
        auc = 0.5

    f1 = f1_score(true_support, selected)
    tp = np.sum(selected & true_support)
    fp = np.sum(selected & ~true_support)
    fdr = fp / max(tp + fp, 1)

    return dict(mse=mse, auc=auc, f1=f1, tp=tp, fp=fp, fdr=fdr, support_size=selected.sum())


# Naive Lasso baseline:


def fit_naive_lasso(X_obs, y_obs, mask, beta_true=None):
    n, p = X_obs.shape

    X_naive = X_obs.copy()

    for j in range(p):
        obs = mask[:, j]
        if obs.sum() > 0:
            X_naive[~obs, j] = X_naive[obs, j].mean()

    scaler = StandardScaler().fit(X_naive)
    X_std = scaler.transform(X_naive)

    y_mean = y_obs.mean()
    y_sd = y_obs.std() + 1e-8
    y_std = (y_obs - y_mean) / y_sd

    model = LassoCV(cv=5, max_iter=100000, tol=1e-6, n_alphas=50).fit(X_std, y_std)
    beta = model.coef_ * y_sd / (scaler.scale_ + 1e-8)
    return beta


# MICE-Lasso baseline:

def fit_mice_lasso(X_obs, y_obs, mask, max_iter_impute=5, n_nearest_features=None):
    n, p = X_obs.shape
    if n_nearest_features is None:
        n_nearest_features = min(30, p - 1) if p > 100 else None

    X_for_impute = X_obs.copy()
    X_for_impute[~mask] = np.nan
    imputer = IterativeImputer(
        estimator=Ridge(alpha=1.0),
        max_iter=max_iter_impute,
        random_state=0,
        sample_posterior=False,
        n_nearest_features=n_nearest_features,
    )
    X_imputed = imputer.fit_transform(X_for_impute)

    scaler = StandardScaler().fit(X_imputed)
    X_std = scaler.transform(X_imputed)
    y_mean, y_sd = y_obs.mean(), y_obs.std() + 1e-8
    y_std = (y_obs - y_mean) / y_sd

    model = LassoCV(cv=5, max_iter=100000, tol=1e-6, n_alphas=50).fit(X_std, y_std)
    beta = model.coef_ * y_sd / (scaler.scale_ + 1e-8)
    return beta


# Adaptive-Huber-Lasso baseline:


def fit_adaptive_huber_lasso(X_obs, mask, y_obs, lam=0.1, delta_const=1.345,
                              max_outer=20, max_iter=500, tol=1e-6):
    n, p = X_obs.shape
    X_naive = X_obs.copy()
    for j in range(p):
        obs = mask[:, j]
        if obs.sum() > 0:
            X_naive[~obs, j] = X_naive[obs, j].mean()

    scaler = StandardScaler().fit(X_naive)
    X_std = scaler.transform(X_naive)
    y_mean, y_sd = y_obs.mean(), y_obs.std() + 1e-8
    y_std = (y_obs - y_mean) / y_sd

    beta = np.zeros(p)
    for _ in range(max_outer):
        resid = y_std - X_std @ beta
        sigma_hat = np.median(np.abs(resid - np.median(resid))) / 0.6745 + 1e-8
        thresh = delta_const * sigma_hat
        w = np.where(np.abs(resid) <= thresh, 1.0, thresh / (np.abs(resid) + 1e-12))

        Xw = X_std * np.sqrt(w)[:, None]
        yw = y_std * np.sqrt(w)
        Sigma_w = (Xw.T @ Xw) / n
        rho_w = (Xw.T @ yw) / n

        beta_new = fit_quadratic_penalty(Sigma_w, rho_w, lam=lam, penalty="lasso",
                                          max_iter=max_iter, tol=tol)
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new

    beta_original_scale = beta * y_sd / (scaler.scale_ + 1e-8)
    return beta_original_scale
