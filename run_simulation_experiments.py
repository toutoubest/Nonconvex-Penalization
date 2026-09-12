import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_generation import generate_data
from missing_contamination import add_missing_contamination
from covariance_estimators import ipw_covariance, robust_ipw_covariance
from shrinkage import lpd_shrinkage_v2, _alpha_min_boundary
from tuning import tune_lambda_ebic
from evaluation import fit_naive_lasso, fit_mice_lasso, fit_adaptive_huber_lasso, evaluate


# Single replicate, all 7 methods


def run_one_custom(seed=1, n=200, p=100, s=10,
                    missing_rate=0.2, contam_rate=0.05, contam_scale=8.0,
                    contamination_type="gaussian", missing_type="MCAR",
                    beta_low=1.0, beta_high=2.0,
                    gamma_ebic=0.0, threshold=1e-3, huber_lam=0.05):

    X, y, beta_true, _ = generate_data(n=n, p=p, s=s, beta_low=beta_low,
                                        beta_high=beta_high, seed=seed)
    X_obs, y_obs, mask = add_missing_contamination(
        X, y, missing_rate=missing_rate, contam_rate=contam_rate,
        contam_scale=contam_scale, contamination_type=contamination_type,
        missing_type=missing_type, seed=seed,
    )

    methods = {}
    methods["Naive-Lasso"] = fit_naive_lasso(X_obs, y_obs, mask)
    methods["MICE-Lasso"] = fit_mice_lasso(X_obs, y_obs, mask)
    methods["Adaptive-Huber-Lasso"] = fit_adaptive_huber_lasso(X_obs, mask, y_obs, lam=huber_lam)

    Sigma_ipw, rho_ipw = ipw_covariance(X_obs, y_obs, mask)
    Sigma_lpd, _, _, _ = lpd_shrinkage_v2(Sigma_ipw, alpha_mode="boundary")
    methods["LPD-Lasso"], _ = tune_lambda_ebic(Sigma_lpd, rho_ipw, n=n, penalty="lasso", gamma_ebic=gamma_ebic)

    Sigma_ripw, rho_ripw = robust_ipw_covariance(X_obs, y_obs, mask)
    Sigma_rlpd, _, _, _ = lpd_shrinkage_v2(Sigma_ripw, alpha_mode="boundary")
    methods["RLPD-Lasso"], _ = tune_lambda_ebic(Sigma_rlpd, rho_ripw, n=n, penalty="lasso", gamma_ebic=gamma_ebic)
    methods["RLPD-SCAD"], _ = tune_lambda_ebic(Sigma_rlpd, rho_ripw, n=n, penalty="scad", gamma_ebic=gamma_ebic)
    methods["RLPD-MCP"], _ = tune_lambda_ebic(Sigma_rlpd, rho_ripw, n=n, penalty="mcp", gamma_ebic=gamma_ebic)

    out = {}
    for name, beta_hat in methods.items():
        out[name] = evaluate(beta_hat, beta_true, threshold=threshold)
    return out


# Repeated simulation wrapper


def run_setting_custom(R=20, **kwargs):
    all_results = {}

    for seed in range(1, R + 1):
        res = run_one_custom(seed=seed, **kwargs)

        for method, metrics in res.items():
            if method not in all_results:
                all_results[method] = []
            all_results[method].append(metrics)

    rows = []
    for method, metrics_list in all_results.items():
        rows.append({
            "Method": method,
            "MSE": np.mean([m["mse"] for m in metrics_list]),
            "AUC": np.mean([m["auc"] for m in metrics_list]),
            "F1": np.mean([m["f1"] for m in metrics_list]),
            "FDR": np.mean([m["fdr"] for m in metrics_list]),
            "TP": np.mean([m["tp"] for m in metrics_list]),
            "FP": np.mean([m["fp"] for m in metrics_list]),
            "SupportSize": np.mean([m["support_size"] for m in metrics_list]),
        })

    return pd.DataFrame(rows)


# Experiment 1: Main comparison (matches "Main comparison under missingness",
# Table tab:sim1: n=200, p=100, s=10, contamination rate 0.05, scale 8,
# missing rate in {0.1, 0.2, 0.3}).

def experiment_1_main(R=20):
    settings = [
        {"missing_rate": 0.1, "contam_rate": 0.05},
        {"missing_rate": 0.2, "contam_rate": 0.05},
        {"missing_rate": 0.3, "contam_rate": 0.05},
        {"missing_rate": 0.2, "contam_rate": 0.10},
    ]

    all_tables = []

    for st in settings:
        df = run_setting_custom(
            R=R,
            n=200, p=100, s=10,
            beta_low=1.0, beta_high=2.0,
            **st
        )
        df.insert(0, "Missing", st["missing_rate"])
        df.insert(1, "Contam", st["contam_rate"])
        all_tables.append(df)

    return pd.concat(all_tables, ignore_index=True)


# Experiment 2: Signal strength (matches Table tab:weak: Strong [1.0, 2.0]
# vs. Weak [0.3, 0.8]; a Moderate setting is kept in between for extra
# resolution).

def experiment_2_weak_signal(R=20):
    settings = [
        {"beta_low": 1.0, "beta_high": 2.0, "Signal": "Strong"},
        {"beta_low": 0.5, "beta_high": 1.0, "Signal": "Moderate"},
        {"beta_low": 0.3, "beta_high": 0.8, "Signal": "Weak"},
    ]

    all_tables = []

    for st in settings:
        df = run_setting_custom(
            R=R,
            n=200, p=100, s=10,
            missing_rate=0.2,
            contam_rate=0.05,
            beta_low=st["beta_low"],
            beta_high=st["beta_high"]
        )
        df.insert(0, "Signal", st["Signal"])
        all_tables.append(df)

    return pd.concat(all_tables, ignore_index=True)


# Experiment 3: High-dimensional p > n.


def experiment_3_high_dimensional(R=20):
    settings = [
        {"n": 200, "p": 100, "s": 10},
        {"n": 200, "p": 500, "s": 25},
        {"n": 200, "p": 1000, "s": 25},
    ]

    all_tables = []

    for st in settings:
        df = run_setting_custom(
            R=R,
            n=st["n"], p=st["p"], s=st["s"],
            missing_rate=0.2,
            contam_rate=0.05,
            beta_low=1.0,
            beta_high=2.0
        )
        df.insert(0, "n", st["n"])
        df.insert(1, "p", st["p"])
        df.insert(2, "s", st["s"])
        all_tables.append(df)

    return pd.concat(all_tables, ignore_index=True)


# Experiment 4a: Contamination type / severity (new; matches Table tab:contam
# and Figure 1(a)-(b)). Crosses 3 contamination types with 3 severities.

def experiment_contamination_types(R=20, n=200, p=100, s=10):
    rows = []
    for ctype in ["gaussian", "t", "leverage"]:
        for scale in [4.0, 8.0, 16.0]:
            df = run_setting_custom(
                R=R, n=n, p=p, s=s, missing_rate=0.2, contam_rate=0.05,
                contam_scale=scale, contamination_type=ctype,
            )
            df.insert(0, "contam_type", ctype)
            df.insert(1, "contam_scale", scale)
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


# Experiment 4b: Missingness mechanism, including MNAR (new; matches Table
# tab:missing and Figure 1(c)-(d)). Crosses MCAR/MNAR with 3 missing rates.

def experiment_missingness_types(R=20, n=200, p=100, s=10):
    rows = []
    for mtype in ["MCAR", "MNAR"]:
        for mr in [0.1, 0.2, 0.3]:
            df = run_setting_custom(
                R=R, n=n, p=p, s=s, missing_rate=mr, contam_rate=0.05,
                missing_type=mtype,
            )
            df.insert(0, "missing_type", mtype)
            df.insert(1, "missing_rate", mr)
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


# Experiment 5: choice of shrinkage intensity alpha (new; matches Table
# tab:alpha and the "Choice of the shrinkage intensity" section). Compares
# the feasibility-boundary alpha (Proposition 1), the Ledoit-Wolf-style
# plug-in alpha (Proposition 2), and a cross-validated alpha.

def select_alpha_cv(X_obs, y_obs, mask, K=5, penalty="mcp", gamma_ebic=0.0,
                     n_grid=10, seed=0):
    n = X_obs.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, K)

    Sigma_full, _ = robust_ipw_covariance(X_obs, y_obs, mask)
    p = Sigma_full.shape[0]
    alpha_boundary, _ = _alpha_min_boundary(Sigma_full)
    grid = np.linspace(alpha_boundary, 1.0, n_grid)

    cv_losses = []
    for alpha in grid:
        fold_losses = []
        for k in range(K):
            val_idx = folds[k]
            train_idx = np.setdiff1d(idx, val_idx)
            Xt, yt, mt = X_obs[train_idx], y_obs[train_idx], mask[train_idx]
            Xv, yv, mv = X_obs[val_idx], y_obs[val_idx], mask[val_idx]

            Sigma_t, rho_t = robust_ipw_covariance(Xt, yt, mt)
            mu_t = max(np.trace(Sigma_t) / p, 1e-4)
            Sigma_lpd_t = alpha * Sigma_t + (1 - alpha) * mu_t * np.eye(p)
            Sigma_lpd_t = (Sigma_lpd_t + Sigma_lpd_t.T) / 2

            beta_hat, _ = tune_lambda_ebic(Sigma_lpd_t, rho_t, n=len(train_idx),
                                            penalty=penalty, gamma_ebic=gamma_ebic, n_lams=15)

            Sigma_v, rho_v = robust_ipw_covariance(Xv, yv, mv)
            val_loss = 0.5 * beta_hat @ Sigma_v @ beta_hat - rho_v @ beta_hat
            fold_losses.append(val_loss)
        cv_losses.append(np.mean(fold_losses))

    best_alpha = grid[int(np.argmin(cv_losses))]
    return best_alpha, alpha_boundary, grid, np.array(cv_losses)


def experiment_alpha_comparison(R=10, n=200, p=100, s=10):
    rows = []
    for seed in range(1, R + 1):
        X, y, beta_true, _ = generate_data(n=n, p=p, s=s, seed=seed)
        X_obs, y_obs, mask = add_missing_contamination(X, y, seed=seed)

        Sigma_ripw, rho_ripw = robust_ipw_covariance(X_obs, y_obs, mask)

        # boundary (Proposition 1)
        Sigma_b, alpha_b, _, _ = lpd_shrinkage_v2(Sigma_ripw, alpha_mode="boundary")
        beta_b, _ = tune_lambda_ebic(Sigma_b, rho_ripw, n=n, penalty="mcp")
        m_b = evaluate(beta_b, beta_true)
        m_b.update(alpha_mode="boundary", alpha=alpha_b, seed=seed)
        rows.append(m_b)

        # Ledoit-Wolf-style plug-in (Proposition 2)
        Sigma_lw, alpha_lw, _, _ = lpd_shrinkage_v2(Sigma_ripw, X_for_lw=X_obs, alpha_mode="lw")
        beta_lw, _ = tune_lambda_ebic(Sigma_lw, rho_ripw, n=n, penalty="mcp")
        m_lw = evaluate(beta_lw, beta_true)
        m_lw.update(alpha_mode="lw", alpha=alpha_lw, seed=seed)
        rows.append(m_lw)

        # Cross-validated alpha (most expensive, but answers "which alpha is
        # actually best" directly rather than via a proxy criterion)
        best_alpha_cv, alpha_boundary, _, _ = select_alpha_cv(X_obs, y_obs, mask, K=3, n_grid=8, seed=seed)
        mu = max(np.trace(Sigma_ripw) / p, 1e-4)
        Sigma_cv = best_alpha_cv * Sigma_ripw + (1 - best_alpha_cv) * mu * np.eye(p)
        beta_cv, _ = tune_lambda_ebic(Sigma_cv, rho_ripw, n=n, penalty="mcp")
        m_cv = evaluate(beta_cv, beta_true)
        m_cv.update(alpha_mode="cv", alpha=best_alpha_cv, seed=seed)
        rows.append(m_cv)

    return pd.DataFrame(rows)


# Plot sensitivity curves 

STYLE = {
    "Naive-Lasso": dict(color="#000000", linestyle="-",  marker="o"),
    "LPD-Lasso":   dict(color="#808080", linestyle="--", marker="s"),
    "RLPD-Lasso":  dict(color="#4A68D9", linestyle=":",  marker="^"),
    "RLPD-SCAD":   dict(color="#CA3142", linestyle="-.", marker="*"),
    "RLPD-MCP":    dict(color="#458933", linestyle="-.", marker="v"),
}
METHOD_ORDER = ["Naive-Lasso", "LPD-Lasso", "RLPD-Lasso", "RLPD-SCAD", "RLPD-MCP"]
MARKERSIZE = {"o": 8, "s": 8, "^": 9, "*": 13, "v": 9}


def _plot_lines(ax, df, xcol, ycol):
    for m in METHOD_ORDER:
        sub = df[df.Method == m].sort_values(xcol)
        st = STYLE[m]
        ax.plot(sub[xcol], sub[ycol],
                label=m, linewidth=2.2,
                color=st["color"], linestyle=st["linestyle"],
                marker=st["marker"], markersize=MARKERSIZE[st["marker"]],
                markerfacecolor=st["color"], markeredgecolor=st["color"])


def save_sensitivity_plots(contam_df, missing_df):
    plt.rcParams.update({
        "font.size": 13,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 150,
    })

    # Fig1a / Fig1b: F1 / mean FP vs. contamination scale, averaged over the
    # 3 contamination types.
    contam_avg = contam_df.groupby(["contam_scale", "Method"])[["F1", "FP"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    _plot_lines(ax, contam_avg, "contam_scale", "F1")
    ax.set_xlabel("Contamination scale")
    ax.set_ylabel("F1 score")
    ax.set_xticks([4, 8, 16])
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig("Fig1a.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    _plot_lines(ax, contam_avg, "contam_scale", "FP")
    ax.set_xlabel("Contamination scale")
    ax.set_ylabel("Mean false positives")
    ax.set_xticks([4, 8, 16])
    fig.tight_layout()
    fig.savefig("Fig1b.pdf")
    plt.close(fig)

    # Fig1c / Fig1d: F1 / mean FP vs. missing rate, MCAR only (MNAR is
    # reported numerically in Table tab:missing instead, to keep this
    # figure's legend at the same 5 methods as Fig1a/1b).
    missing_mcar = missing_df[missing_df["missing_type"] == "MCAR"]

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    _plot_lines(ax, missing_mcar, "missing_rate", "F1")
    ax.set_xlabel("Missing rate")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig("Fig1c.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    _plot_lines(ax, missing_mcar, "missing_rate", "FP")
    ax.set_xlabel("Missing rate")
    ax.set_ylabel("Mean false positives")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig("Fig1d.pdf")
    plt.close(fig)

    print("Saved Fig1a.pdf, Fig1b.pdf, Fig1c.pdf, Fig1d.pdf")


# Run everything.

if __name__ == "__main__":
    exp1_main = experiment_1_main(R=20)
    print("\nExperiment 1: Main comparison")
    print(exp1_main.round(4))
    exp1_main.to_csv("exp1_main_results.csv", index=False)

    exp2_weak = experiment_2_weak_signal(R=20)
    print("\nExperiment 2: Signal strength")
    print(exp2_weak.round(4))
    exp2_weak.to_csv("exp2_weak_signal_results.csv", index=False)

    exp3_hd = experiment_3_high_dimensional(R=20)
    print("\nExperiment 3: High-dimensional p > n")
    print(exp3_hd.round(4))
    exp3_hd.to_csv("exp3_high_dim_results.csv", index=False)

    contam_df = experiment_contamination_types(R=20)
    print("\nExperiment 4a: Contamination type / severity")
    print(contam_df.round(4))
    contam_df.to_csv("contam_results.csv", index=False)

    missing_df = experiment_missingness_types(R=20)
    print("\nExperiment 4b: Missingness mechanism (MCAR / MNAR)")
    print(missing_df.round(4))
    missing_df.to_csv("missing_results.csv", index=False)

    alpha_df = experiment_alpha_comparison(R=10, n=200, p=100, s=10)
    print("\nExperiment 5: Shrinkage-intensity comparison (p=100)")
    print(alpha_df.groupby("alpha_mode")[["f1", "fdr", "auc", "alpha"]].mean())
    alpha_df.to_csv("alpha_comparison_p100.csv", index=False)

    # Plot only after contam_df / missing_df actually exist.
    save_sensitivity_plots(contam_df, missing_df)
