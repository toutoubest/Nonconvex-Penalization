#Optional post-selection thresholding:

def post_threshold(beta, threshold=1e-3):
    beta_new = beta.copy()
    beta_new[np.abs(beta_new) < threshold] = 0.0
    return beta_new



#Evaluation metrics:

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

    return mse, auc, f1, tp, fp



# Naive Lasso baseline:

def fit_naive_lasso(X_obs, y_obs, mask, beta_true=None):
    n, p = X_obs.shape

    X_naive = X_obs.copy()

    for j in range(p):
        obs = mask[:, j]
        if obs.sum() > 0:
            X_naive[~obs, j] = X_naive[obs, j].mean()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_naive)

    y_mean = y_obs.mean()
    y_sd = y_obs.std() + 1e-8
    y_std = (y_obs - y_mean) / y_sd

    alphas = np.logspace(-4, -1, 40)

    best_score = np.inf
    best_beta = None

    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_std, y_std)

        pred = model.predict(X_std)
        s_hat = np.sum(np.abs(model.coef_) > 1e-6)

        score = np.mean((y_std - pred) ** 2) + (s_hat * np.log(n)) / n

        if score < best_score:
            best_score = score
            beta_original = model.coef_ * y_sd / scaler.scale_
            best_beta = beta_original

    return best_beta
