def add_missing_contamination(
    X,
    y,
    missing_rate=0.2,
    contam_rate=0.05,
    contam_scale=8.0,
    seed=1
):
    np.random.seed(seed)

    X_obs = X.copy()
    y_obs = y.copy()

    mask = np.random.rand(*X.shape) > missing_rate
    X_obs[~mask] = 0.0

    n = X.shape[0]
    m = int(contam_rate * n)

    if m > 0:
        contam_idx = np.random.choice(n, m, replace=False)
        X_obs[contam_idx, :] += contam_scale * np.random.normal(size=X_obs[contam_idx, :].shape)
        y_obs[contam_idx] += contam_scale * np.random.normal(size=m)

    return X_obs, y_obs, mask
