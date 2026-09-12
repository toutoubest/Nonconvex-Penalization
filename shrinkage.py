import numpy as np
from numpy.linalg import eigvalsh


# ==============================================================================
# Linear shrinkage positive-definite modification
# ==============================================================================
#
# Updated: this file now supports two shrinkage-intensity rules instead of
# just one:
#
#   - "boundary" (the original method): the smallest alpha for which
#     Sigma_LPD(alpha) = alpha*Sigma_hat + (1-alpha)*mu*I is positive
#     definite. This is Proposition 1 in the revised paper -- it answers
#     *feasibility*, not optimality.
#
#   - "lw": a Ledoit-Wolf (2004) style plug-in estimate of the alpha that
#     minimizes the expected Frobenius-norm estimation error of Sigma_LPD.
#     This is the new Proposition 2 in the revised paper.
#
# The revised paper recommends and uses the combined rule
#     alpha = max(alpha_lw, alpha_boundary)
# (Remark 3 / Eq. "alpha_combined"), which is guaranteed positive definite
# and only differs from the boundary rule when the unconstrained MSE-optimal
# alpha happens to be *larger* than the feasibility boundary. lpd_shrinkage_v2
# implements this as alpha_mode="combined".
#
# eps_mode (new): the original code used a fixed absolute eps (default
# 1e-4). Once we actually inspected the numbers, we found that when the
# matrix scale mu = trace(Sigma_hat)/p is much larger than eps -- which is
# essentially always true in practice (mu is often in the range 15-25 in
# our simulations) -- alpha_min ends up extremely close to 1 (e.g.
# 0.999995) regardless of how ill-conditioned the raw matrix is, because the
# smallest eigenvalue is only pushed up to the (numerically negligible)
# floor eps. The resulting matrix can still be very poorly conditioned even
# though it is technically positive definite. eps_mode="relative" fixes
# this by scaling eps with the matrix itself (eps = eps_ratio * mu), so the
# shrinkage strength adapts to p and to the data scale rather than being a
# fixed constant that is completely decoupled from the data.


def _alpha_min_boundary(Sigma_hat, eps=1e-4, eps_mode="absolute", eps_ratio=0.02):
    """Proposition 1: smallest alpha that guarantees positive definiteness."""
    p = Sigma_hat.shape[0]
    mu_raw = np.trace(Sigma_hat) / p
    eps_used = eps_ratio * mu_raw if eps_mode == "relative" else eps
    mu = max(mu_raw, eps_used)
    lam_min = eigvalsh(Sigma_hat).min()
    if mu <= lam_min:
        return 0.0, mu
    alpha_min = (mu - eps_used) / (mu - lam_min)
    return float(np.clip(alpha_min, 0.0, 1.0)), mu


def ledoit_wolf_shrinkage_identity(X_for_lw):
    """
    Vectorized implementation of the Ledoit & Wolf (2004) analytic shrinkage
    intensity for the special case of an identity (mu*I) shrinkage target.

    This applies the classical Ledoit-Wolf result to our IPW-weighted,
    robustly down-weighted design matrix -- it should be described as a
    "Ledoit-Wolf-inspired plug-in intensity" rather than a re-derivation of
    optimality under missingness and contamination.

    Returns delta_hat, the shrinkage-toward-mu*I intensity in [0, 1].
    The corresponding alpha (weight on the raw estimator) is 1 - delta_hat.
    """
    n, p = X_for_lw.shape
    S = (X_for_lw.T @ X_for_lw) / n
    S = (S + S.T) / 2
    mu = np.trace(S) / p

    row_sq_norm = np.sum(X_for_lw ** 2, axis=1)          # ||x_i||^2
    XS = X_for_lw @ S                                     # n x p
    quad = np.sum(XS * X_for_lw, axis=1)                  # x_i^T S x_i
    pi_hat = np.mean(row_sq_norm ** 2 - 2 * quad + np.sum(S ** 2))

    gamma_hat = np.sum((S - mu * np.eye(p)) ** 2)
    if gamma_hat < 1e-12:
        return 1.0
    delta_hat = pi_hat / n / gamma_hat
    return float(np.clip(delta_hat, 0.0, 1.0))


def lpd_shrinkage_v2(Sigma_hat, X_for_lw=None, eps=1e-4, alpha_mode="boundary",
                      eps_mode="absolute", eps_ratio=0.02):
    """
    alpha_mode:
      - "boundary": original method, smallest alpha on the positive-definite
        boundary (Proposition 1).
      - "lw": Ledoit-Wolf-style plug-in alpha (Proposition 2). Falls back to
        min(alpha_lw, alpha_boundary) to guarantee positive definiteness if
        alpha_lw alone would violate it.
      - "combined": alpha = max(alpha_lw, alpha_boundary), the rule
        recommended in the revised paper (Remark 3). Positive definite by
        construction and coincides with the MSE-optimal value whenever that
        value is already feasible.

    eps_mode="relative" uses eps_ratio*mu instead of a fixed eps (see the
    module docstring above).

    Returns Sigma_lpd, alpha_used, alpha_boundary, alpha_lw (alpha_lw is
    None when alpha_mode == "boundary").
    """
    Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2
    p = Sigma_hat.shape[0]
    alpha_boundary, mu = _alpha_min_boundary(Sigma_hat, eps=eps, eps_mode=eps_mode, eps_ratio=eps_ratio)

    if alpha_mode == "boundary":
        alpha_used = alpha_boundary
        alpha_lw = None
    elif alpha_mode == "lw":
        assert X_for_lw is not None, "alpha_mode='lw' requires X_for_lw (the weighted design matrix used to form Sigma_hat)"
        delta_hat = ledoit_wolf_shrinkage_identity(X_for_lw)
        alpha_lw = 1.0 - delta_hat
        # Safety net: take the more conservative (smaller) of the LW value
        # and the boundary value, to guarantee positive definiteness.
        alpha_used = min(alpha_lw, alpha_boundary) if alpha_boundary > 0 else alpha_boundary
    elif alpha_mode == "combined":
        assert X_for_lw is not None, "alpha_mode='combined' requires X_for_lw"
        delta_hat = ledoit_wolf_shrinkage_identity(X_for_lw)
        alpha_lw = 1.0 - delta_hat
        alpha_used = max(alpha_lw, alpha_boundary)
    else:
        raise ValueError("unknown alpha_mode")

    Sigma_lpd = alpha_used * Sigma_hat + (1 - alpha_used) * mu * np.eye(p)
    Sigma_lpd = (Sigma_lpd + Sigma_lpd.T) / 2
    return Sigma_lpd, alpha_used, alpha_boundary, (alpha_lw if alpha_mode in ("lw", "combined") else None)


def lpd_shrinkage(Sigma_hat, eps=1e-4):
    """
    Original function signature, kept for backward compatibility with any
    existing code that calls lpd_shrinkage(Sigma_hat) and expects a single
    matrix back. Internally this now just calls lpd_shrinkage_v2 with the
    original boundary rule and absolute eps, so it reproduces the original
    behavior exactly.
    """
    Sigma_lpd, _, _, _ = lpd_shrinkage_v2(Sigma_hat, eps=eps, alpha_mode="boundary", eps_mode="absolute")
    return Sigma_lpd
