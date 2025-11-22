import pandas as pd
import ta

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Ensuring all columns are 1D float series
    close = df['adj close'].squeeze().astype(float)
    open = df['open'].squeeze().astype(float)
    # close = df['close'].squeeze().astype(float)
    high  = df['high'].squeeze().astype(float)
    low   = df['low'].squeeze().astype(float)
    vol   = df['volume'].squeeze().astype(float)

    # Core indicators
    out['return_1d'] = close.pct_change()
    out['rsi14'] = ta.momentum.RSIIndicator(close, 14).rsi()
    out['macd_diff'] = ta.trend.MACD(close).macd_diff()
    out['bb_b'] = ta.volatility.BollingerBands(close).bollinger_pband()
    out['atr14'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    out['roc10'] = ta.momentum.ROCIndicator(close, 10).roc()
    out['obv'] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    out['mfi14'] = ta.volume.MFIIndicator(high, low, close, vol, 14).money_flow_index()
    out['willr14'] = ta.momentum.WilliamsRIndicator(high, low, close, 14).williams_r()

    # Prevent lookahead bias
    return out.shift(1)
