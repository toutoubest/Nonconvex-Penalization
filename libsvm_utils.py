import numpy as np


def load_libsvm(path):
    """Minimal LIBSVM-format loader: '<label> <idx>:<val> <idx>:<val> ...' per line.
    Used for the colon-cancer and leukemia (leu / leu.t) gene-expression datasets
    bundled with this repository. Only the design matrix X is returned by the
    dataset-loading helpers in highdim_gene_data.py -- the label column in these
    files is the original tumor/normal or ALL/AML class label, which is discarded
    because the real-data experiments in the paper use a synthetic response
    (see real_data_utils.generate_semisynthetic_from_real_X) built on top of the
    real gene-expression design matrix.
    """
    rows = []
    max_idx = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            d = {}
            for tok in parts[1:]:
                idx, val = tok.split(":")
                idx = int(idx)
                d[idx] = float(val)
                if idx > max_idx:
                    max_idx = idx
            rows.append(d)
    n = len(rows)
    X = np.zeros((n, max_idx))
    for i, d in enumerate(rows):
        for idx, val in d.items():
            X[i, idx - 1] = val
    return X


def variance_screen(X, p_screen):
    """Keep the p_screen columns with the highest raw (pre-standardization)
    variance. Standard practice for gene-expression data where the raw
    feature count (thousands) is computationally infeasible for the full
    EBIC-tuned coordinate descent pipeline used in this paper.

    Screening is applied to the complete, clean design matrix, before any
    missingness or contamination is introduced and without reference to the
    response, so it does not leak outcome information into the subsequent
    variable-selection comparison (see Section 5 of the revised paper)."""
    var = X.var(axis=0)
    top_idx = np.argsort(-var)[:p_screen]
    top_idx = np.sort(top_idx)
    return X[:, top_idx], top_idx
