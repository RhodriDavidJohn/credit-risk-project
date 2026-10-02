import numpy as np
import pandas as pd


def select_columns(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keep only the model's features; anything else in the input is ignored."""
    return X[columns].copy()


def log_loan_amount(X: pd.DataFrame) -> pd.DataFrame:
    """Stateless: replace loan_amnt with its log (nothing learned from data)."""
    X = X.copy()
    X["loan_amnt_log"] = np.log(X["loan_amnt"].clip(lower=1))
    return X.drop(columns=["loan_amnt"])