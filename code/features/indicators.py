# import numpy as np
# import pandas as pd


# # =========================
# # Helper indicator functions
# # =========================

# def rsi(close: pd.Series, window: int = 14) -> pd.Series:
#     delta = close.diff()
#     gain = delta.clip(lower=0.0)
#     loss = -delta.clip(upper=0.0)

#     avg_gain = gain.rolling(window).mean()
#     avg_loss = loss.rolling(window).mean()

#     rs = avg_gain / (avg_loss + 1e-10)
#     return 100.0 - (100.0 / (1.0 + rs))


# def macd_diff(close: pd.Series,
#               fast: int = 12,
#               slow: int = 26,
#               signal: int = 9) -> pd.Series:
#     ema_fast = close.ewm(span=fast, adjust=False).mean()
#     ema_slow = close.ewm(span=slow, adjust=False).mean()
#     macd = ema_fast - ema_slow
#     macd_signal = macd.ewm(span=signal, adjust=False).mean()
#     return macd - macd_signal


# def bb_b(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
#     sma = close.rolling(window).mean()
#     std = close.rolling(window).std()

#     upper = sma + n_std * std
#     lower = sma - n_std * std

#     return (close - lower) / (upper - lower + 1e-10)


# def atr(high: pd.Series,
#         low: pd.Series,
#         close: pd.Series,
#         window: int = 14) -> pd.Series:
#     prev_close = close.shift(1)

#     tr = pd.concat(
#         [
#             high - low,
#             (high - prev_close).abs(),
#             (low - prev_close).abs(),
#         ],
#         axis=1,
#     ).max(axis=1)

#     return tr.rolling(window).mean()


# def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
#     direction = np.sign(close.diff()).fillna(0.0)
#     return (direction * volume).cumsum()


# def mfi(high: pd.Series,
#         low: pd.Series,
#         close: pd.Series,
#         volume: pd.Series,
#         window: int = 14) -> pd.Series:
#     typical_price = (high + low + close) / 3.0
#     money_flow = typical_price * volume

#     direction = np.sign(typical_price.diff())
#     positive_flow = money_flow.where(direction > 0, 0.0)
#     negative_flow = money_flow.where(direction < 0, 0.0)

#     pos_sum = positive_flow.rolling(window).sum()
#     neg_sum = negative_flow.rolling(window).sum()

#     mfr = pos_sum / (neg_sum + 1e-10)
#     return 100.0 - (100.0 / (1.0 + mfr))


# def williams_r(high: pd.Series,
#                low: pd.Series,
#                close: pd.Series,
#                window: int = 14) -> pd.Series:
#     highest_high = high.rolling(window).max()
#     lowest_low = low.rolling(window).min()

#     return -100.0 * (highest_high - close) / (highest_high - lowest_low + 1e-10)


# # =========================
# # Main builder function
# # =========================

# def build_indicators(
#     close: pd.Series,
#     open_: pd.Series,
#     high: pd.Series,
#     low: pd.Series,
#     volume: pd.Series,
#     indicators: list[str],
# ) -> pd.DataFrame:
#     """
#     Build technical indicators specified in config.

#     Parameters
#     ----------
#     close, open_, high, low, volume : pd.Series
#         OHLCV series (1D, float)
#     indicators : list[str]
#         Indicator names from config.yaml

#     Returns
#     -------
#     pd.DataFrame
#         Indicator matrix indexed like `close`
#     """

#     out = pd.DataFrame(index=close.index)

#     for ind in indicators:
#         if ind == "sma10":
#             out["sma10"] = close.rolling(10).mean()

#         elif ind == "sma20":
#             out["sma20"] = close.rolling(20).mean()

#         elif ind == "rsi14":
#             out["rsi14"] = rsi(close, 14)

#         elif ind == "macd_diff":
#             out["macd_diff"] = macd_diff(close)

#         elif ind == "bb_b":
#             out["bb_b"] = bb_b(close)

#         elif ind == "atr14":
#             out["atr14"] = atr(high, low, close, 14)

#         elif ind == "roc10":
#             out["roc10"] = close.pct_change(10)

#         elif ind == "obv":
#             out["obv"] = obv(close, volume)

#         elif ind == "mfi14":
#             out["mfi14"] = mfi(high, low, close, volume, 14)

#         elif ind == "willr14":
#             out["willr14"] = williams_r(high, low, close, 14)

#         else:
#             raise ValueError(f"Unknown indicator: {ind}")

#     return out


import numpy as np
import pandas as pd


# =========================
# Helper indicator functions
# =========================

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-10)
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

    # %B in [0,1] when price is within bands, can go <0 or >1 during extremes
    return (close - lower) / (upper - lower + 1e-10)


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

    mfr = pos_sum / (neg_sum + 1e-10)
    return 100.0 - (100.0 / (1.0 + mfr))


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()

    den = (highest_high - lowest_low)
    out = -100.0 * (highest_high - close) / den.replace(0.0, np.nan)

    return out
    # return -100.0 * (highest_high - close) / (highest_high - lowest_low + 1e-10)


# =========================
# Return / vol primitives
# =========================

def ret(close: pd.Series, k: int = 1) -> pd.Series:
    """
    Simple pct return over k bars.
    Note: returns are stationary-ish and usually very useful for RL.
    """
    return close.pct_change(k)


def vol(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling volatility of daily returns (std of pct returns).
    """
    r = close.pct_change(1)
    return r.rolling(window).std()


# =========================
# Main builder function
# =========================

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

    # Small dispatch table keeps the code tidy as the indicator list grows.
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
