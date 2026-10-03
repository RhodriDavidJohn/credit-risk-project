import numpy as np
import pandas as pd
from collections.abc import Callable
from sklearn.base import BaseEstimator, TransformerMixin



def _fit_bins(x: pd.Series, bins: int = 10) -> np.ndarray | None:
    """Learn quantile edges for numeric variables (None = treat as categorical).
    Outer edges are opened to +/-inf so unseen extremes still land in a bin."""
    if (x.dtype.kind in 'bifc') and (x.nunique()>10):
        _, edges = pd.qcut(x, bins, duplicates='drop', retbins=True)
        edges[0], edges[-1] = -np.inf, np.inf
        return edges
    return None


def _apply_bins(x: pd.Series, edges: np.ndarray | None = None) -> pd.Series:
    """Map raw values to string bin labels, NaN -> 'MISSING'."""
    binned = x if edges is None else pd.cut(x, edges)
    return binned.astype(str).where(x.notna(), "MISSING")


def _bin_variable(x: pd.Series, edges: np.ndarray | None = None) -> pd.Categorical:
    """Bin labels as an ordered categorical, so WoE tables print in natural order."""
    if edges is None:
        order = [str(c) for c in sorted(x.dropna().unique())]
    else:
        order = [str(c) for c in pd.cut(pd.Series([], dtype=float), edges).cat.categories]
    return pd.Categorical(_apply_bins(x, edges), categories=order + ["MISSING"], ordered=True)
        

def _check_binary_target(y: pd.Series) -> None:
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"WoE needs a binary 0/1 target, got {set(y.unique())}")


def _check_non_negative_target(y: pd.Series) -> None:
    if (y < 0).any():
        raise ValueError(f"Continuous WoE needs a non-negative target, got min {y.min()}")
    if y.sum() == 0:
        raise ValueError("Continuous WoE needs at least some positive target values")


def calculate_classification_woe(binned_variable: pd.Series,
                                 target: pd.Series):
    _check_binary_target(target)
    d = (pd.DataFrame({"Cutoff": binned_variable, "y": target})
           .groupby("Cutoff", observed=True)["y"].agg(N="count", Bads="sum")
           .reset_index())
    d["Goods"] = d["N"] - d["Bads"]
    d["% of Bads"] = np.maximum(d["Bads"], 0.5) / d["Bads"].sum()
    d["% of Goods"] = np.maximum(d["Goods"], 0.5) / d["Goods"].sum()
    d["WoE"] = np.log(d["% of Goods"] / d["% of Bads"])
    d["IV"] = (d["% of Goods"] - d["% of Bads"]) * d["WoE"]
    return d


def calculate_continuous_woe(binned_variable: pd.Series,
                             target: pd.Series):
    _check_non_negative_target(target)
    d = (pd.DataFrame({"Cutoff": binned_variable, "y": target})
           .groupby("Cutoff", observed=True)["y"].agg(N="count", Y="sum")
           .reset_index())
    d["% of Obs"] = d["N"] / d["N"].sum()
    d["% of Y"] = np.maximum(d["Y"], 1e-6) / d["Y"].sum()
    d["WoE"] = np.log(d["% of Y"] / d["% of Obs"])
    d["IV"] = (d["% of Y"] - d["% of Obs"]) * d["WoE"]
    return d


def _woe_iv(data: pd.DataFrame,
            target: str,
            woe_func: Callable[[pd.Series, pd.Series], pd.DataFrame],
            bins: int = 10,
            show_woe: bool = True):
    woe_tables = []
    for var in data.columns.drop(target):
        d = woe_func(_bin_variable(data[var], _fit_bins(data[var], bins)), data[target])
        d.insert(0, "Variable", var)
        woe_tables.append(d)
        print(f"Information value of {var} is {d['IV'].sum():.6f}")
        if show_woe:
            print(d)

    woe_df = pd.concat(woe_tables, ignore_index=True)
    iv_df = woe_df.groupby("Variable", sort=False)["IV"].sum().reset_index()
    return iv_df, woe_df



def classification_woe_iv(data: pd.DataFrame,
                          target: str,
                          bins: int = 10,
                          show_woe: bool = True):
    _check_binary_target(data[target])
    return _woe_iv(data, target, calculate_classification_woe, bins, show_woe)


def continuous_woe_iv(data: pd.DataFrame,
                      target: str,
                      bins: int = 10,
                      show_woe: bool = True):
    _check_non_negative_target(data[target])
    return _woe_iv(data, target, calculate_continuous_woe, bins, show_woe)


class WoETransformer(BaseEstimator, TransformerMixin):
    """Learns bins + WoE values on training data, then applies them to any data.

    fit(X, y)    -> learns bin edges and a {bin label: WoE} map per variable
    transform(X) -> replaces each variable with '<var>_woe'; other columns pass through
    """

    def __init__(self,
                 variables: list[str] | None = None,
                 bins: int = 10,
                 woe_func: Callable[[pd.Series, pd.Series], pd.DataFrame] = calculate_classification_woe,
                 unseen_woe: float = 0.0,
                 custom_bins: dict[str, list[float]] | None = None):
        self.variables = variables      # None = encode every column
        self.bins = bins
        self.woe_func = woe_func
        self.unseen_woe = unseen_woe    # WoE for a bin never seen in training (0 = average risk)
        self.custom_bins = custom_bins

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = pd.Series(np.asarray(y), index=X.index)   # align by position, not by old index labels
        self.variables_ = list(X.columns) if self.variables is None else list(self.variables)
        self.edges_, self.woe_maps_, self.woe_tables_ = {}, {}, {}

        for var in self.variables_:
            if self.custom_bins and var in self.custom_bins:
                edges = np.array([-np.inf, *sorted(self.custom_bins[var]), np.inf])
            else:
                edges = _fit_bins(X[var], self.bins)
            table = self.woe_func(_bin_variable(X[var], edges), y)
            self.edges_[var] = edges
            self.woe_maps_[var] = dict(zip(table["Cutoff"].astype(str), table["WoE"]))
            self.woe_tables_[var] = table
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for var in self.variables_:
            labels = _apply_bins(X[var], self.edges_[var])
            X[f"{var}_woe"] = labels.map(self.woe_maps_[var]).fillna(self.unseen_woe)
        return X.drop(columns=self.variables_)
