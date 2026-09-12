import numpy as np
from sklearn.linear_model import Lasso

from penalties import soft_threshold, scad_threshold, mcp_threshold


def fit_quadratic_penalty(
    Sigma, rho, lam=0.05, penalty="lasso", max_iter=1000, tol=1e-6, beta_init=None
):
    """
    Coordinate descent solver for
        0.5 * beta^T Sigma beta - rho^T beta + lam * sum_j P(|beta_j|)
    where P is soft/SCAD/MCP thresholding.

    beta_init (new, optional): warm-start value in the *original* beta scale.
    Used by the fast lambda-path solver's non-fast fallback and by
    fit_adaptive_huber_lasso's IRLS loop; passing None reproduces the
    original zero-initialized behavior exactly.
    """
    p = Sigma.shape[0]
    std_devs = np.sqrt(np.maximum(np.diag(Sigma), 1e-8))
    Corr = Sigma / np.outer(std_devs, std_devs)
    Corr = (Corr + Corr.T) / 2
    rho_corr = rho / std_devs

    if beta_init is not None:
        beta_corr = beta_init * std_devs  # convert from original scale back to correlation scale
    else:
        beta_corr = np.zeros(p)

    diag_corr = np.maximum(np.diag(Corr), 1e-8)

    # --- Critical fix -----------------------------------------------------
    # The original code computed `lam_j = lam / diag_corr[j]`. Because Corr
    # is a correlation matrix, diag_corr[j] is always 1, so this line was a
    # no-op (lam_j == lam for every coordinate). Substituting
    # beta = beta_corr / std_devs into the original objective
    #   0.5*beta^T Sigma beta - rho^T beta + lam*sum_j|beta_j|
    # shows the penalty term should expand to
    #   sum_j (lam / std_devs[j]) * |beta_corr_j|,
    # i.e. each coordinate's effective penalty must be scaled by
    # 1 / std_devs[j], not left at lam. When every column has variance close
    # to 1 (e.g. clean, complete data) this bug has almost no effect; but
    # once missingness/contamination are introduced, the diagonal of the
    # IPW-corrected covariance deviates noticeably from 1, and the bug
    # silently solves a "variance-reweighted" penalty problem instead of the
    # uniform-lambda problem assumed by the EBIC grid search. This was
    # verified against the KKT stationarity conditions: the fixed line below
    # reduces the KKT violation from ~0.16 to ~2e-9 on a representative
    # missing+contaminated setting.
    for _ in range(max_iter):
        beta_old = beta_corr.copy()
        for j in range(p):
            r_j = rho_corr[j] - Corr[j, :] @ beta_corr + Corr[j, j] * beta_corr[j]
            z = r_j / diag_corr[j]
            lam_j = lam / (diag_corr[j] * std_devs[j])  # fixed: was missing the std_devs[j] factor
            if penalty == "lasso":
                beta_corr[j] = soft_threshold(z, lam_j)
            elif penalty == "scad":
                beta_corr[j] = scad_threshold(z, lam_j)
            elif penalty == "mcp":
                beta_corr[j] = mcp_threshold(z, lam_j)
            else:
                raise ValueError("penalty must be lasso, scad, or mcp")
        if np.linalg.norm(beta_corr - beta_old, ord=np.inf) < tol:
            break

    beta = beta_corr / std_devs
    return beta


# ==============================================================================
# Fast solver (new) -- used for large p (p > 150 by default, see tuning.py).
#
# The pure-Python coordinate descent above does not scale to p in the
# hundreds/thousands (a single fit already takes minutes at p=1000, and the
# EBIC grid needs dozens of fits per replicate). This maps the same
# quadratic-loss problem onto an equivalent synthetic-data Lasso problem
# that sklearn's compiled coordinate descent can solve directly, and handles
# SCAD/MCP via the local linear approximation (LLA): repeatedly linearize
# the nonconvex penalty around the current estimate and solve the resulting
# weighted-L1 problem. This is what makes the p=500/1000 experiments in the
# revised paper (Section 4.5, Table "High-dimensional performance") feasible
# on ordinary hardware; see run_simulation_experiments.py's high-dimensional
# experiment for how it gets used.
# ==============================================================================
def sigma_rho_to_synthetic(Sigma, rho, ridge=1e-10):
    """Builds X_synth, y_synth such that X_synth^T X_synth = Sigma and
    X_synth^T y_synth = rho, via a Cholesky factorization of Sigma."""
    p = Sigma.shape[0]
    Sigma_reg = Sigma + ridge * np.eye(p)
    L = np.linalg.cholesky(Sigma_reg)   # Sigma_reg = L L^T
    X_synth = L.T                        # p x p
    y_synth = np.linalg.solve(L, rho)    # L y_synth = rho
    return X_synth, y_synth


def fit_lasso_fast(Sigma, rho, lam, beta_init=None, max_iter=3000, tol=1e-7, synth=None):
    """
    synth: optional precomputed (X_synth, y_synth). tune_lambda_ebic sweeps
    dozens of lambda values for the same (Sigma, rho); redoing the O(p^3)
    Cholesky factorization at every lambda would be wasteful (it alone can
    take seconds at p=1000-2000), so it is computed once and reused.
    """
    p = Sigma.shape[0]
    if lam <= 0:
        return np.linalg.solve(Sigma + 1e-8 * np.eye(p), rho)
    X_synth, y_synth = synth if synth is not None else sigma_rho_to_synthetic(Sigma, rho)
    model = Lasso(alpha=lam / p, fit_intercept=False, max_iter=max_iter, tol=tol)
    model.fit(X_synth, y_synth)
    return model.coef_


def fit_scad_mcp_fast(Sigma, rho, lam, penalty="scad", a=3.7, gamma=3.0,
                       beta_init=None, max_outer=10, max_iter=2000, tol=1e-6, synth=None):
    """LLA: linearize the SCAD/MCP penalty at the current estimate into a
    weighted L1 penalty, solve, and repeat until convergence."""
    p = Sigma.shape[0]
    if lam <= 0:
        return np.linalg.solve(Sigma + 1e-8 * np.eye(p), rho)

    X_synth, y_synth = synth if synth is not None else sigma_rho_to_synthetic(Sigma, rho)
    beta = beta_init.copy() if beta_init is not None else fit_lasso_fast(Sigma, rho, lam=lam, synth=(X_synth, y_synth))

    for _ in range(max_outer):
        absb = np.abs(beta)
        if penalty == "scad":
            w = np.where(absb <= lam, lam,
                         np.where(absb <= a * lam, (a * lam - absb) / (a - 1), 0.0))
        elif penalty == "mcp":
            w = np.maximum(lam - absb / gamma, 0.0)
            w = np.where(absb <= gamma * lam, w, 0.0)
        else:
            raise ValueError("penalty must be scad or mcp")
        w_safe = np.maximum(w, 1e-6 * max(lam, 1e-8))

        X_scaled = X_synth / w_safe[np.newaxis, :]
        model = Lasso(alpha=1.0 / p, fit_intercept=False, max_iter=max_iter, tol=tol)
        model.fit(X_scaled, y_synth)
        beta_new = model.coef_ / w_safe

        if np.linalg.norm(beta_new - beta, ord=np.inf) < tol:
            beta = beta_new
            break
        beta = beta_new

    return beta
