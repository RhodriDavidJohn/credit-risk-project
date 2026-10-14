import numpy as np
import pandas as pd


def psi_from_shares(expected: pd.Series, actual: pd.Series, floor: float = 1e-4) -> float:
    """PSI between two distributions given as {bin: share}; bins missing from one side get `floor`."""
    e, a = expected.align(actual, fill_value=0)
    e, a = np.maximum(e, floor), np.maximum(a, floor)
    return float(((a - e) * np.log(a / e)).sum())


def characteristic_stability(woe_transformer, X_expected: pd.DataFrame, X_actual: pd.DataFrame) -> pd.Series:
    """CSI: a PSI for each model feature, on the WoE bins the model actually uses."""
    bins_e = woe_transformer.bin_labels(X_expected)
    bins_a = woe_transformer.bin_labels(X_actual)
    return pd.Series({var: psi_from_shares(bins_e[var].value_counts(normalize=True),
                                           bins_a[var].value_counts(normalize=True))
                      for var in bins_e.columns}).sort_values(ascending=False)


def expected_psi_noise(n_bins: int, n_expected: int, n_actual: int) -> float:
    """Approximate PSI from sampling noise alone, when both samples come from the SAME population."""
    return (n_bins - 1) * (1 / n_expected + 1 / n_actual)