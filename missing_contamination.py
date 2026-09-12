import numpy as np


def add_missing_contamination(
    X,
    y,
    missing_rate=0.2,
    contam_rate=0.05,
    contam_scale=8.0,
    contamination_type="gaussian",   # "gaussian" | "t" | "leverage"
    missing_type="MCAR",             # "MCAR" | "MNAR"
    seed=1
):
    """
    Missingness + contamination generator.

    contamination_type:
      - "gaussian": original mechanism. A contam_rate fraction of rows gets
        N(0, contam_scale^2) noise added to both X and y.
      - "t": heavy-tailed contamination using a Student-t (df=3) noise term,
        closer to the classical heavy-tailed setting used in robust statistics.
      - "leverage": only inflates the norm of the contaminated rows of X
        (constructing leverage points); y is left untouched. This isolates
        whether robust down-weighting at the covariance level helps, since
        the response itself carries no contamination signal here.

    missing_type:
      - "MCAR": original mechanism, elementwise-independent random missingness.
      - "MNAR": missingness probability depends on the covariate's own value.
        For each column, values above its 75th percentile are more likely to
        be missing, values below are less likely, while the overall rate
        still averages out to roughly missing_rate.
    """
    rng = np.random.RandomState(seed)

    X_obs = X.copy()
    y_obs = y.copy()
    n, p = X.shape

    if missing_type == "MCAR":
        mask = rng.rand(n, p) > missing_rate
    elif missing_type == "MNAR":
        q75 = np.percentile(X, 75, axis=0)
        high_val = X > q75
        p_high = min(missing_rate * 2.0, 0.95)
        p_low = missing_rate * 0.5
        miss_prob = np.where(high_val, p_high, p_low)
        mask = rng.rand(n, p) > miss_prob
    else:
        raise ValueError("unknown missing_type")

    X_obs[~mask] = 0.0

    m = int(contam_rate * n)

    if m > 0:
        contam_idx = rng.choice(n, m, replace=False)

        if contamination_type == "gaussian":
            X_obs[contam_idx, :] += contam_scale * rng.normal(size=X_obs[contam_idx, :].shape)
            y_obs[contam_idx] += contam_scale * rng.normal(size=m)
        elif contamination_type == "t":
            X_obs[contam_idx, :] += contam_scale * rng.standard_t(df=3, size=X_obs[contam_idx, :].shape)
            y_obs[contam_idx] += contam_scale * rng.standard_t(df=3, size=m)
        elif contamination_type == "leverage":
            direction = rng.normal(size=(m, p))
            direction /= (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12)
            row_norm = np.linalg.norm(X_obs[contam_idx, :], axis=1, keepdims=True) + 1e-8
            X_obs[contam_idx, :] += contam_scale * direction * row_norm
            # y is intentionally left unchanged, to build pure X-direction leverage points
        else:
            raise ValueError("unknown contamination_type")

    return X_obs, y_obs, mask
