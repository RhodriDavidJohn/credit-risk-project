import numpy as np
import pandas as pd
from scipy.stats import chi2, binomtest
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, brier_score_loss, log_loss



def gini(y_true, y_pred) -> float:
    return 2 * roc_auc_score(y_true, y_pred) - 1


def ks_statistic(y_true, y_pred) -> tuple[float, float]:
    """KS statistic and the PD threshold where it occurs."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    i = np.argmax(tpr - fpr)
    return tpr[i] - fpr[i], thresholds[i]


def hosmer_lemeshow_test(y_true, y_pred, n_groups: int = 10, groups=None) -> tuple[float, int, float]:
    """HL statistic, degrees of freedom, p-value. Groups = PD quantiles unless given (e.g. rating grades)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)   # positional, never index-aligned
    if groups is None:
        groups = pd.qcut(y_pred, q=n_groups, labels=False, duplicates="drop")
    d = (pd.DataFrame({"y": y_true, "p": y_pred, "g": np.asarray(groups)})
           .groupby("g").agg(n=("y", "size"), observed=("y", "sum"), expected=("p", "sum")))
    sq = (d["observed"] - d["expected"]) ** 2
    hl = (sq / d["expected"] + sq / (d["n"] - d["expected"])).sum()   # defaults + non-defaults
    dof = len(d) - 2                                                  # groups actually formed, not requested
    return hl, dof, chi2.sf(hl, dof)


def binomial_calibration_test(y_true, y_prob, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Exact binomial test of observed vs predicted default rate in each PD bin."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    bin_ids = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)

    results = []
    for i in range(n_bins):
        mask = bin_ids == i
        n = mask.sum()
        if n == 0:
            continue
        k = int(y_true[mask].sum())
        p_pred = y_prob[mask].mean()
        p_val = binomtest(k, n, p=np.clip(p_pred, 1e-6, 1 - 1e-6)).pvalue
        results.append({"bin": i, "total_n": n, "observed_pos": k,
                        "expected_prob": p_pred, "observed_prob": k / n, "p_value": p_val})
    return pd.DataFrame(results)


def evaluate_pd(y_true, y_pred) -> pd.Series:
    """One-stop summary: discrimination + calibration for a set of PD predictions."""
    ks, ks_threshold = ks_statistic(y_true, y_pred)
    hl, dof, hl_p = hosmer_lemeshow_test(y_true, y_pred)
    return pd.Series({
        "Gini": gini(y_true, y_pred),
        "KS": ks,
        "KS threshold": ks_threshold,
        "Average precision": average_precision_score(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_pred),
        "Log loss": log_loss(y_true, y_pred),
        "Mean PD": np.mean(y_pred),
        "Default rate": np.mean(y_true),
        "HL": hl,
        "HL p": hl_p,
    })


def psi(expected, actual, n_bins: int = 10) -> pd.DataFrame:
    """Population Stability Index table; bins are quantiles of `expected` (the reference population)."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    edges = np.unique(np.percentile(expected, np.linspace(0, 100, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    exp_pct = np.histogram(expected, edges)[0] / len(expected)
    act_pct = np.histogram(actual, edges)[0] / len(actual)
    exp_pct, act_pct = np.maximum(exp_pct, 1e-4), np.maximum(act_pct, 1e-4)
    return pd.DataFrame({"Lower": edges[:-1], "Upper": edges[1:],
                         "Expected %": exp_pct * 100, "Actual %": act_pct * 100,
                         "PSI": (act_pct - exp_pct) * np.log(act_pct / exp_pct)})


# ---------- the model: fit once, predict many times ----------

class ProbabilityOfDefault:
    """A PD model = an estimator (e.g. a Pipeline) plus an optional calibration step.

    Data is passed to methods, not stored, so one fitted model can be evaluated on any dataset.
    """

    def __init__(self, model, calibration: str | None = "isotonic", cv: int = 5):
        if calibration not in (None, "isotonic", "sigmoid"):
            raise ValueError("calibration must be None, 'isotonic' or 'sigmoid'")
        self.model = model
        self.calibration = calibration
        self.cv = cv

    def _build(self):
        """A fresh, unfitted copy, so the caller's model object is never modified."""
        if self.calibration is None:
            return clone(self.model)
        return CalibratedClassifierCV(clone(self.model), method=self.calibration, cv=self.cv)

    def fit(self, X, y):
        self.fitted_model_ = self._build().fit(X, y)
        return self

    def predict_pd(self, X) -> np.ndarray:
        if not hasattr(self, "fitted_model_"):
            raise RuntimeError("Call fit() before predict_pd()")
        return self.fitted_model_.predict_proba(X)[:, 1]

    def evaluate(self, X, y) -> pd.Series:
        return evaluate_pd(y, self.predict_pd(X))

    def calibration_table(self, X, y, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
        return binomial_calibration_test(y, self.predict_pd(X), n_bins, strategy)

    def out_of_fold_pd(self, X, y, n_splits: int = 5, random_state: int = 0) -> np.ndarray:
        """Honest PDs for every training row: each comes from a model (incl. calibration) that never saw it."""
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
        return cross_val_predict(self._build(), X, y, cv=cv, method="predict_proba")[:, 1]