import numpy as np
import pandas as pd

from real_data_utils import load_real_datasets, preprocess_X, generate_semisynthetic_from_real_X
from missing_contamination import add_missing_contamination
from covariance_estimators import ipw_covariance, robust_ipw_covariance
from shrinkage import lpd_shrinkage_v2
from tuning import tune_lambda_ebic
from evaluation import post_threshold, evaluate, fit_naive_lasso, fit_mice_lasso, fit_adaptive_huber_lasso


# Run one semi-synthetic real-data experiment on a real design matrix X.
#
# Uses all 7 methods compared throughout the paper (Naive-Lasso, MICE-Lasso,
# Adaptive-Huber-Lasso, LPD-Lasso, RLPD-Lasso, RLPD-SCAD, RLPD-MCP) and
# lpd_shrinkage_v2 (alpha_mode="boundary"), the same shrinkage rule used
# everywhere else in the paper.

def run_one_real_dataset(
    X,
    seed=1,
    s=15,
    missing_rate=0.2,
    contam_rate=0.05,
    contam_scale=8.0,
    beta_low=1.0,
    beta_high=2.0,
    sigma=1.0,
    gamma_ebic=0.0,
    threshold=1e-3,
    huber_lam=0.05
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

    methods = {}

    # Naive-Lasso
    methods["Naive-Lasso"] = fit_naive_lasso(X_obs, y_obs, mask)

    # MICE-Lasso
    methods["MICE-Lasso"] = fit_mice_lasso(X_obs, y_obs, mask)

    # Adaptive-Huber-Lasso
    methods["Adaptive-Huber-Lasso"] = fit_adaptive_huber_lasso(X_obs, mask, y_obs, lam=huber_lam)

    # LPD-Lasso
    Sigma_ipw, rho_ipw = ipw_covariance(X_obs, y_obs, mask)
    Sigma_lpd, _, _, _ = lpd_shrinkage_v2(Sigma_ipw, alpha_mode="boundary")
    beta_lpd_lasso, _ = tune_lambda_ebic(
        Sigma_lpd, rho_ipw, n=n, penalty="lasso", gamma_ebic=gamma_ebic
    )
    methods["LPD-Lasso"] = beta_lpd_lasso

    # Robust LPD methods
    Sigma_ripw, rho_ripw = robust_ipw_covariance(X_obs, y_obs, mask)
    Sigma_rlpd, _, _, _ = lpd_shrinkage_v2(Sigma_ripw, alpha_mode="boundary")

    beta_rlpd_lasso, _ = tune_lambda_ebic(
        Sigma_rlpd, rho_ripw, n=n, penalty="lasso", gamma_ebic=gamma_ebic
    )
    beta_rlpd_scad, _ = tune_lambda_ebic(
        Sigma_rlpd, rho_ripw, n=n, penalty="scad", gamma_ebic=gamma_ebic
    )
    beta_rlpd_mcp, _ = tune_lambda_ebic(
        Sigma_rlpd, rho_ripw, n=n, penalty="mcp", gamma_ebic=gamma_ebic
    )
    methods["RLPD-Lasso"] = beta_rlpd_lasso
    methods["RLPD-SCAD"] = beta_rlpd_scad
    methods["RLPD-MCP"] = beta_rlpd_mcp

    results = {}
    for name, beta_hat in methods.items():
        beta_hat = post_threshold(beta_hat, threshold=threshold)
        results[name] = evaluate(beta_hat, beta_true, threshold=threshold)

    return results, n, p, int(np.sum(beta_true != 0))


# Repeated real-data experiment for one dataset:

def run_real_dataset_repeated(
    X,
    dataset_name,
    R=20,
    s=15,
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

        for method, metrics in res.items():
            if method not in all_results:
                all_results[method] = []
            all_results[method].append(metrics)

    rows = []

    for method, metrics_list in all_results.items():
        mse_mean = np.mean([m["mse"] for m in metrics_list])
        auc_mean = np.mean([m["auc"] for m in metrics_list])
        f1_mean = np.mean([m["f1"] for m in metrics_list])
        fdr_mean = np.mean([m["fdr"] for m in metrics_list])
        tp_mean = np.mean([m["tp"] for m in metrics_list])
        fp_mean = np.mean([m["fp"] for m in metrics_list])
        support_mean = np.mean([m["support_size"] for m in metrics_list])

        rows.append({
            "Dataset": dataset_name,
            "n": n_final,
            "p": p_final,
            "s": s_final,
            "Method": method,
            "MSE": mse_mean,
            "AUC": auc_mean,
            "F1": f1_mean,
            "FDR": fdr_mean,
            "TP": tp_mean,
            "FP": fp_mean,
            "SupportSize": support_mean
        })

    return pd.DataFrame(rows)


# Run all real datasets.
#
# R=10 and s=15 match Table "tab:real" in the paper. p_screen is forwarded to
# real_data_utils.load_real_datasets(); pass a smaller value for a much
# faster (but not paper-matching) smoke test.

def run_all_real_datasets(
    R=10,
    missing_rate=0.2,
    contam_rate=0.05,
    contam_scale=8.0,
    s=15,
    p_screen=300
):
    datasets = load_real_datasets(p_screen=p_screen)

    all_tables = []

    for name, X in datasets.items():
        print(f"\n{name} raw shape: {X.shape}")  # sanity check, e.g. Colon should print (62, 300)
        print(f"Running real-data experiment: {name}")

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


if __name__ == "__main__":
    # Reproduces Table "tab:real" in the paper: Colon Cancer, Leukemia, and
    # Riboflavin, each variance-screened to p=300, with R=10 replications
    # and s=15.
    real_results = run_all_real_datasets()
    print("\nSemi-synthetic real data results:")
    print(real_results.round(4))
    real_results.to_csv("real_highdim_results.csv", index=False)
