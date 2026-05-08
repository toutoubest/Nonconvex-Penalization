#Linear shrinkage positive definite modification

def lpd_shrinkage(Sigma_hat, eps=1e-4):
    Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2
    lam_min = eigvalsh(Sigma_hat).min()

    if lam_min > eps:
        return Sigma_hat

    mu = max(np.trace(Sigma_hat) / Sigma_hat.shape[0], eps)
    alpha = (mu - eps) / (mu - lam_min)

    Sigma_lpd = alpha * Sigma_hat + (1 - alpha) * mu * np.eye(Sigma_hat.shape[0])
    Sigma_lpd = (Sigma_lpd + Sigma_lpd.T) / 2

    return Sigma_lpd
