import numpy as np


def load_libsvm(path):
    
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
    
    var = X.var(axis=0)
    top_idx = np.argsort(-var)[:p_screen]
    top_idx = np.sort(top_idx)
    return X[:, top_idx], top_idx
