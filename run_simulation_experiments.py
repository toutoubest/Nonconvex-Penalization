# Repeated simulation wrapper

def run_setting_custom(R=20, **kwargs):
    all_results = {}

    for seed in range(1, R + 1):
        res = run_one_custom(seed=seed, **kwargs)

        for method, values in res.items():
            if method not in all_results:
                all_results[method] = []
            all_results[method].append(values)

    rows = []
    for method, values in all_results.items():
        arr = np.array(values)
        mean_vals = arr.mean(axis=0)

        rows.append({
            "Method": method,
            "MSE": mean_vals[0],
            "AUC": mean_vals[1],
            "F1": mean_vals[2],
            "TP": mean_vals[3],
            "FP": mean_vals[4]
        })

    return pd.DataFrame(rows)



# Experiment 1: Main comparison:

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



# Experiment 2: Weak signal setting:

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



# Experiment 3: High-dimensional p > n:

def experiment_3_high_dimensional(R=20):
    settings = [
        {"n": 200, "p": 200, "s": 10},
        {"n": 200, "p": 500, "s": 10},
        {"n": 200, "p": 500, "s": 20},
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


# Experiment 4: Sensitivity curves:

def experiment_4_sensitivity(R=20):
    contam_grid = [0.00, 0.05, 0.10, 0.15, 0.20]
    missing_grid = [0.10, 0.20, 0.30, 0.40, 0.50]

    contam_tables = []
    for cr in contam_grid:
        df = run_setting_custom(
            R=R,
            n=200, p=100, s=10,
            missing_rate=0.2,
            contam_rate=cr,
            beta_low=1.0,
            beta_high=2.0
        )
        df.insert(0, "Contam", cr)
        contam_tables.append(df)

    missing_tables = []
    for mr in missing_grid:
        df = run_setting_custom(
            R=R,
            n=200, p=100, s=10,
            missing_rate=mr,
            contam_rate=0.05,
            beta_low=1.0,
            beta_high=2.0
        )
        df.insert(0, "Missing", mr)
        missing_tables.append(df)

    contam_df = pd.concat(contam_tables, ignore_index=True)
    missing_df = pd.concat(missing_tables, ignore_index=True)

    return contam_df, missing_df



# Plot sensitivity curves

import matplotlib.pyplot as plt


# Line style + color settings

line_styles = {

    # Baselines (gray/black)
    'Naive-Lasso': ('solid', 'o', 'black'),
    'LPD-Lasso': ('dashed', 's', 'gray'),
    # Proposed methods
    'RLPD-Lasso': ('dotted', '^', 'royalblue'),
    'RLPD-SCAD': ('dashdot', 'd', 'crimson'),
    'RLPD-MCP': ((0, (3, 5, 1, 5)), 'v', 'forestgreen')
}

def save_refined_plots(contam_df, missing_df):

    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'xtick.labelsize': 20,
        'ytick.labelsize': 20
    })

    
    # F1 vs Contamination:
    
    plt.figure(figsize=(7, 5))

    for method in contam_df["Method"].unique():
        sub = contam_df[contam_df["Method"] == method]

        ls, mk, clr = line_styles.get(
            method, ('solid', 'o', 'black')
        )

        plt.plot(
            sub["Contam"],
            sub["F1"],
            linestyle=ls,
            marker=mk,
            color=clr,
            linewidth=2.5,
            markersize=7,
            label=method
        )

    plt.xlabel("Contamination rate")
    plt.ylabel("F1 score")

    plt.legend(
        fontsize=11,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("1Fig1a.pdf", bbox_inches='tight')
    plt.show()

    
    # FP vs Contamination:
    
    plt.figure(figsize=(7, 5))

    for method in contam_df["Method"].unique():
        sub = contam_df[contam_df["Method"] == method]

        ls, mk, clr = line_styles.get(
            method, ('solid', 'o', 'black')
        )

        plt.plot(
            sub["Contam"],
            sub["FP"],
            linestyle=ls,
            marker=mk,
            color=clr,
            linewidth=2.5,
            markersize=7,
            label=method
        )

    plt.xlabel("Contamination rate")
    plt.ylabel("False positives")

    plt.legend(
        fontsize=11,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("1Fig1b.pdf", bbox_inches='tight')
    plt.show()

    
    # F1 vs Missingness:
    
    plt.figure(figsize=(7, 5))

    for method in missing_df["Method"].unique():
        sub = missing_df[missing_df["Method"] == method]

        ls, mk, clr = line_styles.get(
            method, ('solid', 'o', 'black')
        )

        plt.plot(
            sub["Missing"],
            sub["F1"],
            linestyle=ls,
            marker=mk,
            color=clr,
            linewidth=2.5,
            markersize=7,
            label=method
        )

    plt.xlabel("Missing rate")
    plt.ylabel("F1 score")

    plt.legend(
        fontsize=11,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("1Fig1c.pdf", bbox_inches='tight')
    plt.show()

    
    # FP vs Missingness:
    
    plt.figure(figsize=(7, 5))

    for method in missing_df["Method"].unique():
        sub = missing_df[missing_df["Method"] == method]

        ls, mk, clr = line_styles.get(
            method, ('solid', 'o', 'black')
        )

        plt.plot(
            sub["Missing"],
            sub["FP"],
            linestyle=ls,
            marker=mk,
            color=clr,
            linewidth=2.5,
            markersize=7,
            label=method
        )

    plt.xlabel("Missing rate")
    plt.ylabel("False positives")

    plt.legend(
        fontsize=11,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("1Fig1d.pdf", bbox_inches='tight')
    plt.show()


# Run plotting function
save_refined_plots(contam_sens, missing_sens)

# Run all experiments:

exp1_main = experiment_1_main(R=20)
print("\nExperiment 1: Main comparison")
print(exp1_main.round(4))

exp2_weak = experiment_2_weak_signal(R=20)
print("\nExperiment 2: Weak signal")
print(exp2_weak.round(4))

exp3_hd = experiment_3_high_dimensional(R=20)
print("\nExperiment 3: High-dimensional p > n")
print(exp3_hd.round(4))

contam_sens, missing_sens = experiment_4_sensitivity(R=20)
print("\nExperiment 4A: Contamination sensitivity")
print(contam_sens.round(4))

print("\nExperiment 4B: Missingness sensitivity")
print(missing_sens.round(4))

