import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score


def vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factor per column (1 = independent of the others; >5 = concern)."""
    return pd.Series(np.diag(np.linalg.inv(X.corr().values)), index=X.columns)


def compare_models(models: dict, X: pd.DataFrame, y: pd.Series, cv, baseline: str) -> pd.DataFrame:
    """Score every model on the SAME folds and report paired differences against `baseline`."""
    gini = {name: 2 * cross_val_score(m, X, y, cv=cv, scoring="roc_auc") - 1
            for name, m in models.items()}
    rows = []
    for name, g in gini.items():
        diff = g - gini[baseline]
        rows.append({"Model": name,
                     "Gini": g.mean(),
                     "Gini sd": g.std(),
                     f"Δ vs {baseline}": diff.mean(),
                     "Δ sd": diff.std(),
                     "% folds better": np.nan if name == baseline else (diff > 0).mean() * 100})
    return pd.DataFrame(rows).set_index("Model")


def coefficient_checks(pipe, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Fit pipe and report coefficients, expected-sign check (WoE -> negative) and VIF."""
    pipe.fit(X, y)
    Z = pipe[:-1].transform(X)
    coefs = pd.Series(pipe[-1].coef_[0], index=Z.columns)
    return pd.DataFrame({"Coefficient": coefs,
                         "Sign OK": [c < 0 if col.endswith("_woe") else np.nan for col, c in coefs.items()],
                         "VIF": vif(Z)})