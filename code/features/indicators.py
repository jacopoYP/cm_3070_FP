import numpy as np
import pandas as pd

from core.math_utils import safe_divide

# =========================
# Helper indicator functions
# =========================

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    # rs = avg_gain / (avg_loss + 1e-10)
    rs = safe_divide(avg_gain, avg_loss, fill_value=np.inf)
    return 100.0 - (100.0 / (1.0 + rs))


def macd_diff(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal


def bb_b(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()

    upper = sma + n_std * std
    lower = sma - n_std * std

    band_width = upper - lower
    normalized = safe_divide(close - lower, band_width)

    return normalized

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    money_flow = typical_price * volume

    direction = np.sign(typical_price.diff())
    positive_flow = money_flow.where(direction > 0, 0.0)
    negative_flow = money_flow.where(direction < 0, 0.0)

    pos_sum = positive_flow.rolling(window).sum()
    neg_sum = negative_flow.rolling(window).sum()

    # mfr = pos_sum / (neg_sum + 1e-10)
    mfr = safe_divide(pos_sum, neg_sum, fill_value=np.inf)
    return 100.0 - (100.0 / (1.0 + mfr))


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()

    den = (highest_high - lowest_low)
    out = -100.0 * (highest_high - close) / den.replace(0.0, np.nan)

    return out

# ---------------------------------------------------------------------
# Return / vol primitives
# ---------------------------------------------------------------------

def ret(close: pd.Series, k: int = 1) -> pd.Series:
    # Simple pct return over k bars.
    return close.pct_change(k)


def vol(close: pd.Series, window: int = 20) -> pd.Series:
    # Rolling volatility of daily returns (std of pct returns).
    
    r = close.pct_change(1)
    return r.rolling(window).std()


# ---------------------------------------------------------------------
# Main builder function
# ---------------------------------------------------------------------

def build_indicators(
    close: pd.Series,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    indicators: list[str],
) -> pd.DataFrame:
    """
    Build technical indicators specified in config.

    (I intentionally produce NaNs during warm-up periods (rolling windows).
    The feature pipeline will drop NaNs after scaling so all tickers align in a clean way.)
    """
    out = pd.DataFrame(index=close.index)

    # Each lambda returns a pd.Series aligned to `close.index`.
    compute = {
        "sma10":      lambda: close.rolling(10).mean(),
        "sma20":      lambda: close.rolling(20).mean(),
        "rsi14":      lambda: rsi(close, 14),
        "macd_diff":  lambda: macd_diff(close),
        "bb_b":       lambda: bb_b(close),
        "atr14":      lambda: atr(high, low, close, 14),
        "roc10":      lambda: close.pct_change(10),
        "obv":        lambda: obv(close, volume),
        "mfi14":      lambda: mfi(high, low, close, volume, 14),
        "willr14":    lambda: williams_r(high, low, close, 14),

        # “prod-ish” primitives (recommended)
        "ret_1":      lambda: ret(close, 1),
        "ret_5":      lambda: ret(close, 5),
        "vol_20":     lambda: vol(close, 20),
    }

    for ind in indicators:
        fn = compute.get(ind)
        if fn is None:
            raise ValueError(f"Unknown indicator: {ind}")
        out[ind] = fn()

    return out
