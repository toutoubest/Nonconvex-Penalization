 #Run one semi-synthetic real-data experiment:

def run_one_real_dataset(
    X,
    seed=1,
    s=10,
    missing_rate=0.2,
    contam_rate=0.05,
    contam_scale=8.0,
    beta_low=1.0,
    beta_high=2.0,
    sigma=1.0,
    gamma_ebic=0.0,
    threshold=1e-3
):
    X = preprocess_X(X)
    n, p = X.shape

    y, beta_true = generate_semisynthetic_from_real_X(
        X,
        s=s,
        sigma=sigma,
        beta_low=beta_low,
        beta_high=beta_high,
        seed=seed
    )

    X_obs, y_obs, mask = add_missing_contamination(
        X,
        y,
        missing_rate=missing_rate,
        contam_rate=contam_rate,
        contam_scale=contam_scale,
        seed=seed
    )

    # Naive-Lasso
    beta_naive = fit_naive_lasso(X_obs, y_obs, mask)

    # LPD-Lasso
    Sigma_ipw, rho_ipw = ipw_covariance(X_obs, y_obs, mask)
    Sigma_lpd = lpd_shrinkage(Sigma_ipw)

    beta_lpd_lasso, _ = tune_lambda_ebic(
        Sigma_lpd,
        rho_ipw,
        n=n,
        penalty="lasso",
        gamma_ebic=gamma_ebic
    )

    # Robust LPD methods
    Sigma_ripw, rho_ripw = robust_ipw_covariance(X_obs, y_obs, mask)
    Sigma_rlpd = lpd_shrinkage(Sigma_ripw)

    beta_rlpd_lasso, _ = tune_lambda_ebic(
        Sigma_rlpd,
        rho_ripw,
        n=n,
        penalty="lasso",
        gamma_ebic=gamma_ebic
    )

    beta_rlpd_scad, _ = tune_lambda_ebic(
        Sigma_rlpd,
        rho_ripw,
        n=n,
        penalty="scad",
        gamma_ebic=gamma_ebic
    )

    beta_rlpd_mcp, _ = tune_lambda_ebic(
        Sigma_rlpd,
        rho_ripw,
        n=n,
        penalty="mcp",
        gamma_ebic=gamma_ebic
    )

    methods = {
        "Naive-Lasso": beta_naive,
        "LPD-Lasso": beta_lpd_lasso,
        "RLPD-Lasso": beta_rlpd_lasso,
        "RLPD-SCAD": beta_rlpd_scad,
        "RLPD-MCP": beta_rlpd_mcp
    }

    results = {}
    for name, beta_hat in methods.items():
        beta_hat = post_threshold(beta_hat, threshold=threshold)
        results[name] = evaluate(beta_hat, beta_true, threshold=threshold)

    return results, n, p, np.sum(beta_true != 0)


# Repeated real-data experiment for one dataset:

def run_real_dataset_repeated(
    X,
    dataset_name,
    R=20,
    s=10,
    missing_rate=0.2,
    contam_rate=0.05,
    contam_scale=8.0,
    beta_low=1.0,
    beta_high=2.0
):
    all_results = {}
    n_final, p_final, s_final = None, None, None

    for seed in range(1, R + 1):
        res, n_final, p_final, s_final = run_one_real_dataset(
            X,
            seed=seed,
            s=s,
            missing_rate=missing_rate,
            contam_rate=contam_rate,
            contam_scale=contam_scale,
            beta_low=beta_low,
            beta_high=beta_high
        )

        for method, values in res.items():
            if method not in all_results:
                all_results[method] = []
            all_results[method].append(values)

    rows = []

    for method, values in all_results.items():
        arr = np.array(values)
        mean_vals = arr.mean(axis=0)

        rows.append({
            "Dataset": dataset_name,
            "n": n_final,
            "p": p_final,
            "s": s_final,
            "Method": method,
            "MSE": mean_vals[0],
            "AUC": mean_vals[1],
            "F1": mean_vals[2],
            "TP": mean_vals[3],
            "FP": mean_vals[4]
        })

    return pd.DataFrame(rows)



# Run all  real datasets

def run_all_real_datasets(
    R=20,
    missing_rate=0.2,
    contam_rate=0.05,
    contam_scale=8.0,
    s=10
):
    datasets = load_real_datasets()

    all_tables = []

    for name, X in datasets.items():
        print(f"\nRunning real-data experiment: {name}")

        df = run_real_dataset_repeated(
            X,
            dataset_name=name,
            R=R,
            s=s,
            missing_rate=missing_rate,
            contam_rate=contam_rate,
            contam_scale=contam_scale
        )

        all_tables.append(df)

    final_df = pd.concat(all_tables, ignore_index=True)

    return final_df



# Run real data experiments

real_results = run_all_real_datasets(
    R=20,
    missing_rate=0.2,
    contam_rate=0.05,
    contam_scale=8.0,
    s=10
)

print("\nSemi-synthetic real data results:")
print(real_results.round(4))

