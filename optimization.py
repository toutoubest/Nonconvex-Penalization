#Coordinate descent solver

def fit_quadratic_penalty(
    Sigma,
    rho,
    lam=0.05,
    penalty="lasso",
    max_iter=1000,
    tol=1e-6
):
    p = Sigma.shape[0]

    std_devs = np.sqrt(np.maximum(np.diag(Sigma), 1e-8))
    Corr = Sigma / np.outer(std_devs, std_devs)
    Corr = (Corr + Corr.T) / 2
    rho_corr = rho / std_devs

    beta_corr = np.zeros(p)
    diag_corr = np.maximum(np.diag(Corr), 1e-8)

    for _ in range(max_iter):
        beta_old = beta_corr.copy()

        for j in range(p):
            r_j = rho_corr[j] - Corr[j, :] @ beta_corr + Corr[j, j] * beta_corr[j]
            z = r_j / diag_corr[j]
            lam_j = lam / diag_corr[j]

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
