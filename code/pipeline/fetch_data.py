import yfinance as yf
import pandas as pd

def fetch_ohlcv(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data using yfinance and clean column names.
    """
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False)
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index)
    df = df[['open', 'high', 'low', 'close', 'adj close', 'volume']].dropna()
    return df