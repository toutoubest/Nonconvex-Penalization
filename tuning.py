#EBIC tuning with fixes

def tune_lambda_ebic(
    Sigma,
    rho,
    n,
    penalty="lasso",
    gamma_ebic=0.0,
    n_lams=40,
    min_ratio=1e-4,
    max_ratio=0.5
):
    p = Sigma.shape[0]

    std_devs = np.sqrt(np.maximum(np.diag(Sigma), 1e-8))
    rho_corr = rho / std_devs

    lam_max = np.max(np.abs(rho_corr))

    if lam_max <= 1e-10:
        return np.zeros(p), 0.0

    lams = np.logspace(
        np.log10(lam_max * min_ratio),
        np.log10(lam_max * max_ratio),
        n_lams
    )

    best_score = np.inf
    best_beta = None
    best_lam = None

    for lam in lams:
        beta = fit_quadratic_penalty(Sigma, rho, lam=lam, penalty=penalty)
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
        best_beta = fit_quadratic_penalty(Sigma, rho, lam=fallback_lam, penalty=penalty)
        best_lam = fallback_lam

    return best_beta, best_lam
