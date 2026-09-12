import numpy as np
import pandas as pd

from libsvm_utils import load_libsvm




def load_colon_cancer(path="colon-cancer"):
    """Colon Cancer dataset (Alon et al., 1999): n=62 tissue samples,
    p=2000 genes, LIBSVM format."""
    X = load_libsvm(path)
    assert X.shape == (62, 2000), f"unexpected colon-cancer shape {X.shape}"
    return X


def load_leukemia_combined(train_path="leu", test_path="leu.t"):
    """Combine the LIBSVM leu (train, n=38) and leu.t (test, n=34) splits into
    the full Golub et al. (1999) leukemia design matrix (n=72, p=7129). Class
    labels are ignored -- only the real gene-expression matrix X is used."""
    Xtr = load_libsvm(train_path)
    Xte = load_libsvm(test_path)
    assert Xtr.shape[1] == Xte.shape[1], f"feature count mismatch: {Xtr.shape} vs {Xte.shape}"
    X = np.vstack([Xtr, Xte])
    assert X.shape == (72, 7129), f"unexpected combined leukemia shape {X.shape}"
    return X


def load_riboflavin(path="riboflavin.csv"):
    """Load riboflavin.csv (Buhlmann, Kalisch & Meier, 2014; obtained from
    R's hdi package via write.csv(cbind(y, x), ..., row.names=FALSE)). First
    column is the response y (not used here ,we generate a semi-synthetic
    response for the support-recovery comparison, same as Colon/Leukemia);
    the remaining 4088 columns are the gene-expression predictors."""
    df = pd.read_csv(path)
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"unexpected non-numeric columns in {path}: {non_numeric}")
    assert df.columns[0] == "y", f"expected first column 'y', got {df.columns[0]!r}"
    y = df["y"].to_numpy(dtype=float)
    X = df.drop(columns=["y"]).to_numpy(dtype=float)
    assert X.shape == (71, 4088), f"unexpected riboflavin X shape {X.shape}"
    return X, y


def load_highdim_gene_datasets(colon_path="colon-cancer", leu_train_path="leu",
                                leu_test_path="leu.t", riboflavin_path="riboflavin.csv"):
    """Convenience wrapper returning all three raw (unscreened) design
    matrices as a dict, keyed the same way as the paper's Table (Section 5):
    {"Colon": (62, 2000), "Leukemia": (72, 7129), "Riboflavin": (71, 4088)}."""
    Xr, _ = load_riboflavin(riboflavin_path)
    return {
        "Colon": load_colon_cancer(colon_path),
        "Leukemia": load_leukemia_combined(leu_train_path, leu_test_path),
        "Riboflavin": Xr,
    }


if __name__ == "__main__":
    datasets = load_highdim_gene_datasets()
    for name, X in datasets.items():
        print(f"{name}: {X.shape}")
