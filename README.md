# Robust High-Dimensional Regression under Missingness and Contamination via Nonconvex Penalization

The proposed framework combines:

- Inverse probability weighting (IPW)
- Linear shrinkage positive-definite (LPD) covariance estimation
- Robust weighting for contamination resistance
- Nonconvex penalization using SCAD and MCP

to perform stable and robust variable selection in high-dimensional regression with incomplete and contaminated data.

---

# Repository Structure

```text
Nonconvex-Penalization/
│
├── covariance_estimators.py
├── data_generation.py
├── evaluation.py
├── missing_contamination.py
├── optimization.py
├── penalties.py
├── real_data_utils.py
├── run_real_data_experiments.py
├── run_simulation_experiments.py
├── shrinkage.py
├── tuning.py
└── README.md
```

---

# File Description

## `data_generation.py`

Contains functions for generating synthetic high-dimensional regression data.

Main functions:

- `generate_data()`

---

## `missing_contamination.py`

Implements missingness and contamination mechanisms.

Main functions:

- `add_missing_contamination()`

---

## `covariance_estimators.py`

Contains covariance and cross-moment estimators.

Main functions:

- `ipw_covariance()`
- `robust_ipw_covariance()`

---

## `shrinkage.py`

Implements linear shrinkage positive-definite modification.

Main functions:

- `lpd_shrinkage()`

---

## `penalties.py`

Contains threshold operators for penalized estimation.

Main functions:

- `soft_threshold()`
- `scad_threshold()`
- `mcp_threshold()`

---

## `optimization.py`

Implements coordinate descent optimization for quadratic penalized regression.

Main functions:

- `fit_quadratic_penalty()`

---

## `tuning.py`

Implements EBIC-based tuning parameter selection.

Main functions:

- `tune_lambda_ebic()`

---

## `evaluation.py`

Contains model evaluation utilities and baseline methods.

Main functions:

- `evaluate()`
- `post_threshold()`
- `fit_naive_lasso()`

---

## `real_data_utils.py`

Contains helper functions for loading and preprocessing real datasets.

---

## `run_simulation_experiments.py`

Runs all simulation experiments in the paper, including:

- Main comparison experiments
- Weak signal experiments
- High-dimensional experiments
- Sensitivity analysis

This script also generates tables and figures.

---

## `run_real_data_experiments.py`

Runs semi-synthetic real data experiments on:

- Diabetes dataset
- Wine dataset
- Liver Disorders dataset

---

# Installation

Clone the repository:

```bash
git clone https://github.com/toutoubest/Nonconvex-Penalization.git
cd Nonconvex-Penalization
```

Install required packages:

```bash
pip install numpy scipy pandas scikit-learn matplotlib
```

---

# Running Simulation Experiments

Run:

```bash
python run_simulation_experiments.py
```

This script reproduces the simulation studies and sensitivity analyses reported in the paper.

---

# Running Real Data Experiments

Run:

```bash
python run_real_data_experiments.py
```

This script reproduces the semi-synthetic real data experiments.

---

# Methods Included

The following methods are implemented:

- Naive-Lasso
- LPD-Lasso
- RLPD-Lasso
- RLPD-SCAD
- RLPD-MCP

---

# Evaluation Metrics

Performance is evaluated using:

- Mean Squared Error (MSE)
- Area Under the ROC Curve (AUC)
- F1 Score
- True Positives (TP)
- False Positives (FP)

---

# Reproducibility

All experiments are conducted with fixed random seeds for reproducibility.

---

# Citation

If you use this code, please cite:

```text
Robust High-Dimensional Regression under Missingness and Contamination via Nonconvex Penalization
```

---

# License

This project is released for academic research purposes.
