import numpy as np

from optimization import fit_quadratic_penalty, sigma_rho_to_synthetic, fit_lasso_fast, fit_scad_mcp_fast


def tune_lambda_ebic(
    Sigma,
    rho,
    n,
    penalty="lasso",
    gamma_ebic=0.0,
    n_lams=40,
    min_ratio=1e-4,
    max_ratio=0.5,
    fast=None,
    scad_mcp_max_outer=None
):
    """
    Updated: added the `fast` / `scad_mcp_max_outer` options so this scales to
    the high-dimensional settings used in the revised paper (p up to 1000).

    fast=None (default): uses the original pure-Python coordinate descent for
    p<=150 (unchanged behavior for existing small-p experiments), and
    automatically switches to the Cholesky + sklearn-Lasso fast solver in
    optimization.py for p>150 -- the pure-Python solver becomes impractically
    slow well before p=1000 (a single fit can take minutes). Both solvers
    have been cross-checked to agree on their optimum; you can also force
    fast=True/False explicitly.

    For large p, the lambda grid and the number of SCAD/MCP LLA outer
    iterations are shrunk automatically to keep runtime reasonable (p=1000 is
    the slowest case, since every lambda on the grid needs several LLA
    rounds, each a full sklearn Lasso fit). If you have more compute budget
    and want a finer grid, pass n_lams / scad_mcp_max_outer explicitly.
    """
    p = Sigma.shape[0]
    if fast is None:
        fast = p > 150
    if fast and p > 500 and n_lams == 40:
        n_lams = 15  # halve the default lambda grid for very large p; warm starts usually make up for it
    if scad_mcp_max_outer is None:
        scad_mcp_max_outer = 10 if p <= 500 else 5

    std_devs = np.sqrt(np.maximum(np.diag(Sigma), 1e-8))
    rho_corr = rho / std_devs

    # lam_max must be computed on the same scale as the `lam` argument that
    # gets passed into the solvers below, i.e. against the *raw* rho, not
    # rho_corr. fit_quadratic_penalty's per-coordinate threshold is
    # lam / std_devs[j], compared against rho_corr[j] = rho[j] / std_devs[j];
    # the std_devs[j] factors cancel, so the all-zero-solution boundary is
    # |rho[j]| <= lam, not |rho_corr[j]| <= lam. The fast solver path
    # (fit_lasso_fast/fit_scad_mcp_fast) matches this: it builds synthetic
    # data with X_synth^T y_synth = rho (unstandardized) and its sklearn
    # Lasso KKT zero-condition is |rho[j]| <= lam as well. Using
    # max(|rho_corr|) here therefore understates lam_max whenever
    # std_devs > 1 (e.g. std_devs ~ 16-47 in the p=100 main simulation
    # setting), by exactly that factor -- which silently truncates the
    # lambda grid well below the EBIC-optimal region and causes systematic
    # over-selection. See the root-cause investigation into the Table 1 vs.
    # Table 2 discrepancy in the revised paper.
    lam_max = np.max(np.abs(rho))

    if lam_max <= 1e-10:
        return np.zeros(p), 0.0

    # Sorted from large to small lambda, with warm starts: the sparse
    # solution at a large lambda is used as the starting point for the next
    # (smaller) lambda, which converges much faster, especially for large p.
    lams = np.logspace(
        np.log10(lam_max * max_ratio),
        np.log10(lam_max * min_ratio),
        n_lams
    )

    best_score = np.inf
    best_beta = None
    best_lam = None
    beta_warm = None

    # The Cholesky factorization is done once and shared across the whole
    # lambda grid -- avoids repeating an O(p^3) factorization at every lambda.
    synth = sigma_rho_to_synthetic(Sigma, rho) if fast else None

    for lam in lams:
        if fast:
            if penalty == "lasso":
                beta = fit_lasso_fast(Sigma, rho, lam=lam, beta_init=beta_warm, synth=synth)
            else:
                beta = fit_scad_mcp_fast(Sigma, rho, lam=lam, penalty=penalty, beta_init=beta_warm,
                                          synth=synth, max_outer=scad_mcp_max_outer)
        else:
            beta = fit_quadratic_penalty(Sigma, rho, lam=lam, penalty=penalty, beta_init=beta_warm)
        beta_warm = beta

        s_hat = np.sum(np.abs(beta) > 1e-6)

        if s_hat == 0:
            continue

        if s_hat >= n:
            continue

        quad_loss = 0.5 * beta.T @ Sigma @ beta - rho.T @ beta

        ebic = quad_loss + (
            s_hat * np.log(n) + 2 * gamma_ebic * s_hat * np.log(p)
        ) / n

        if ebic < best_score:
            best_score = ebic
            best_beta = beta
            best_lam = lam

    if best_beta is None:
        fallback_lam = lam_max * min_ratio
        if fast:
            best_beta = (fit_lasso_fast(Sigma, rho, lam=fallback_lam, synth=synth) if penalty == "lasso"
                         else fit_scad_mcp_fast(Sigma, rho, lam=fallback_lam, penalty=penalty, synth=synth,
                                                 max_outer=scad_mcp_max_outer))
        else:
            best_beta = fit_quadratic_penalty(Sigma, rho, lam=fallback_lam, penalty=penalty)
        best_lam = fallback_lam

    return best_beta, best_lam
