import numpy as np
import pandas as pd

MAX_AGE = 100
MIN_WORKING_AGE = 14


def apply_data_quality_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic validity rules: invalid values -> NaN. Same rules in training and production."""
    df = df.copy()
    df.loc[df["person_age"] > MAX_AGE, "person_age"] = np.nan
    invalid_emp = df["person_emp_length"] > df["person_age"] - MIN_WORKING_AGE
    df.loc[invalid_emp, "person_emp_length"] = np.nan
    return df


def data_quality_report(df: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "age > 100":             (df["person_age"] > MAX_AGE).sum(),
        "emp_length > age - 14": (df["person_emp_length"] > df["person_age"] - MIN_WORKING_AGE).sum(),
        "duplicate rows":        df.duplicated().sum(),
    })