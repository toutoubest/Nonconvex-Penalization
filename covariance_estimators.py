#Robust IPW covariance and cross moment
def robust_ipw_covariance(X_obs, y_obs, mask, c_factor=1.345):
    n, p = X_obs.shape

    row_norms = np.linalg.norm(X_obs, axis=1)
    med = np.median(row_norms)
    mad = np.median(np.abs(row_norms - med))

    c = c_factor * (mad / 0.6745) if mad > 1e-8 else 3.0
    c = max(c, 1.0)

    w = np.minimum(1.0, c / (row_norms + 1e-8))

    Xw = X_obs * np.sqrt(w[:, None])

    pi_xx = (mask.T @ mask) / n
    pi_xx = np.maximum(pi_xx, 1e-4)

    Sigma_ripw = ((Xw.T @ Xw) / n) / pi_xx
    Sigma_ripw = (Sigma_ripw + Sigma_ripw.T) / 2

    pi_xy = mask.mean(axis=0)
    pi_xy = np.maximum(pi_xy, 1e-4)

    rho_ripw = ((X_obs.T @ (w * y_obs)) / n) / pi_xy

    return Sigma_ripw, rho_ripw

 #Standard IPW covariance and cross moment:

def ipw_covariance(X_obs, y_obs, mask):
    n, p = X_obs.shape

    pi_xx = (mask.T @ mask) / n
    pi_xx = np.maximum(pi_xx, 1e-4)

    Sigma_ipw = ((X_obs.T @ X_obs) / n) / pi_xx
    Sigma_ipw = (Sigma_ipw + Sigma_ipw.T) / 2

    pi_xy = mask.mean(axis=0)
    pi_xy = np.maximum(pi_xy, 1e-4)

    rho_ipw = ((X_obs.T @ y_obs) / n) / pi_xy

    return Sigma_ipw, rho_ipw
