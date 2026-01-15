import pandas as pd

def rolling_zscore(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Rolling z-score normalization to avoid data leakage.
    """
    return (df - df.rolling(window).mean()) / df.rolling(window).std()

def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Cap extreme values to reduce impact of outliers.
    """
    return df.clip(lower=df.quantile(lower), upper=df.quantile(upper), axis=1)
