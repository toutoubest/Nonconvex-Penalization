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
#
# Updated: evaluate() now returns a dict instead of a 5-tuple, and adds two
# fields (fdr, support_size) that the revised paper reports throughout
# Section 4 (every table there includes FDR and support size, not just
# MSE/AUC/F1/TP/FP). run_simulation_experiments.py and
# run_real_data_experiments.py have both been updated to consume this dict
# instead of unpacking a tuple positionally.

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
#
# Updated: the original implementation did a manual grid search over 40
# alpha values with a fixed-lambda Lasso, which produced a large number of
# ConvergenceWarnings and is generally less reliable than a proper
# cross-validated path solver. Replaced with LassoCV (a compiled path
# solver with a much higher default iteration budget), which is what the
# revised paper's Naive-Lasso baseline actually uses. beta_true is kept as
# an accepted (unused) keyword for backward compatibility with any existing
# call sites that pass it.

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


# MICE-Lasso baseline (new):
#
# A stronger missing-data baseline than mean imputation. Uses
# IterativeImputer (a single-chain, MICE-style imputation -- not a full
# multiple-imputation-with-Rubin's-rule combination; if you want that, run
# IterativeImputer several times with different random seeds and average /
# pool the resulting fits) instead of mean imputation, followed by LassoCV.
#
# Speed note: IterativeImputer defaults to fitting a BayesianRidge model per
# feature, which already takes 8+ seconds per imputation at p=100. Switching
# the inner regressor to Ridge (with sample_posterior=False, since we don't
# need the Bayesian posterior draws) is roughly 10-15x faster; the bigger
# lever is n_nearest_features -- capping it (e.g. to 30) instead of using
# all p-1 other columns cuts a single imputation at p=500 from ~102s to ~2s
# with no visible quality loss in our AR(1)-correlated design (only a
# limited number of neighboring columns actually carry information). This
# is what makes it feasible to include this baseline at larger p.

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


# Adaptive-Huber-Lasso baseline (new):
#
# Adaptive Huber regression (Sun, Zhou & Fan, 2020 style) plus an L1
# penalty, solved via IRLS (each step re-weights observations using a Huber
# threshold, then does a weighted-least-squares-with-L1 fit using the
# existing fit_quadratic_penalty coordinate descent solver). No extra
# dependency is needed.
#
# Note: this baseline operates on the mean-imputed complete matrix (it does
# not apply the IPW covariance correction of covariance_estimators.py) -- it
# is a "robust loss, no missingness correction" ablation, meant to show how
# much is lost by using a robust loss alone without correcting for
# missingness at the covariance level.

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
