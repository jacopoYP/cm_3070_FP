from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# If you run this script directly (python scripts/build_features.py), add project root to sys.path.
# If you run as module (recommended): python -m clean_trading_rl.scripts.build_features ... , this isn't needed.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import load_config
from features.indicators import build_indicators


def _to_lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


# def _pick_price_columns(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
#     """
#     Accepts yfinance output with columns like:
#       Open, High, Low, Close, Adj Close, Volume
#     and returns lowercase series:
#       close (adj close preferred), open, high, low, volume
#     """
#     df = _to_lower_cols(df)

#     # yfinance uses "adj close" with space
#     if "adj close" in df.columns:
#         close = df["adj close"].squeeze().astype(float)
#     elif "close" in df.columns:
#         close = df["close"].squeeze().astype(float)
#     else:
#         raise ValueError(f"Missing close/adj close column. Available: {list(df.columns)}")

#     required = ["open", "high", "low", "volume"]
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing columns {missing}. Available: {list(df.columns)}")

#     open_ = df["open"].squeeze().astype(float)
#     high = df["high"].squeeze().astype(float)
#     low = df["low"].squeeze().astype(float)
#     vol = df["volume"].squeeze().astype(float)

#     return close, open_, high, low, vol


def _rolling_zscore(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling z-score scaling. Uses only past data (no look-ahead).
    """
    mu = x.rolling(window=window, min_periods=window).mean()
    sigma = x.rolling(window=window, min_periods=window).std(ddof=0)
    return (x - mu) / (sigma.replace(0.0, np.nan))


def download_ohlcv(tickers, start_date, end_date, interval) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",   # MultiIndex columns: (field, ticker)
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError("No data returned. Check tickers/date range.")
    return df

def build_features_for_all_tickers(cfg, raw: pd.DataFrame):
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns from yfinance when downloading multiple tickers.")

    indicators = list(cfg.features.technical_indicators)

    scaling_cfg = cfg.features.scaling or {}
    scaling_method = scaling_cfg.get("method")
    scaling_window = scaling_cfg.get("window")

    all_features = []
    all_prices = []
    debug_rows = []

    for t in cfg.data.tickers:
        # Slice one ticker from MultiIndex columns: (field, ticker)
        df_t = raw.xs(t, axis=1, level=1, drop_level=True)  # columns become ['Adj Close','Close','High','Low','Open','Volume']

        # Standardize column names to lower
        df_t.columns = [str(c).strip().lower() for c in df_t.columns]

        # Pick OHLCV (Adj Close preferred)
        if "adj close" in df_t.columns:
            close = df_t["adj close"].astype(float)
        elif "close" in df_t.columns:
            close = df_t["close"].astype(float)
        else:
            raise ValueError(f"{t}: missing 'adj close'/'close' in columns {list(df_t.columns)}")

        open_ = df_t["open"].astype(float)
        high  = df_t["high"].astype(float)
        low   = df_t["low"].astype(float)
        vol   = df_t["volume"].astype(float)

        feats = build_indicators(
            close=close,
            open_=open_,
            high=high,
            low=low,
            volume=vol,
            indicators=indicators,
        )

        # Optional scaling
        if scaling_method is not None:
            if scaling_method == "rolling_zscore":
                if scaling_window is None:
                    raise ValueError("scaling.window must be set for rolling_zscore")
                feats = _rolling_zscore(feats, int(scaling_window))
            else:
                raise ValueError(f"Unsupported scaling.method: {scaling_method}")

        feats = feats.dropna()
        px = close.loc[feats.index].astype(float)

        all_features.append(feats)
        all_prices.append(px)

        debug_rows.append({
            "ticker": t,
            "rows_raw": int(len(df_t)),
            "rows_features": int(len(feats)),
            "feature_dim": int(feats.shape[1]),
            "first_date": str(feats.index.min()),
            "last_date": str(feats.index.max()),
        })

    features_df = pd.concat(all_features, axis=0)
    prices_s = pd.concat(all_prices, axis=0)

    if len(features_df) != len(prices_s):
        raise RuntimeError(f"Alignment mismatch: features={len(features_df)} prices={len(prices_s)}")

    return features_df.values.astype("float32"), prices_s.values.astype("float32"), pd.DataFrame(debug_rows)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="data")
    p.add_argument("--save_csv", action="store_true", help="Also save features.csv and prices.csv for debugging")
    args = p.parse_args()

    cfg = load_config(args.config)

    # Your config is currently flat (tickers/start_date/...) based on your YAML snippet
    # Keep it simple: use those fields directly.
    # tickers = list(cfg.tickers)
    # start_date = str(cfg.start_date)
    # end_date = str(cfg.end_date)
    # interval = str(cfg.interval)
    # Support both nested (cfg.data.*) and flat (cfg.*) config styles
    data_cfg = getattr(cfg, "data", cfg)

    tickers = list(getattr(data_cfg, "tickers"))
    start_date = str(getattr(data_cfg, "start_date"))
    end_date = str(getattr(data_cfg, "end_date"))
    interval = str(getattr(data_cfg, "interval"))


    os.makedirs(args.out, exist_ok=True)

    print("=== BUILD FEATURES ===")
    print("config:", args.config)
    print("out:", args.out)
    print("tickers:", tickers)
    print("range:", start_date, "→", end_date, "interval:", interval)
    print("indicators:", list(cfg.features.technical_indicators))
    print("scaling:", cfg.features.scaling)


    # Step 1: download
    raw = download_ohlcv(tickers, start_date, end_date, interval)
    print("downloaded rows:", len(raw), "cols:", list(raw.columns))

    # Step 2: build features
    features, prices, dbg = build_features_for_all_tickers(cfg, raw)

    # Step 3: validate
    print("\n=== VALIDATION ===")
    print(dbg.to_string(index=False))
    print("features shape:", features.shape)
    print("prices shape:", prices.shape)
    print("features dtype:", features.dtype, "prices dtype:", prices.dtype)

    if not np.isfinite(features).all():
        raise RuntimeError("features contains NaN/Inf after dropna - check scaling window / indicator math.")
    if not np.isfinite(prices).all():
        raise RuntimeError("prices contains NaN/Inf - unexpected.")

    # Step 4: save artifacts
    f_path = os.path.join(args.out, "features.npy")
    p_path = os.path.join(args.out, "prices.npy")
    d_path = os.path.join(args.out, "build_debug.csv")

    np.save(f_path, features)
    np.save(p_path, prices)
    dbg.to_csv(d_path, index=False)

    if args.save_csv:
        # These can be large — only use for debugging
        pd.DataFrame(features).to_csv(os.path.join(args.out, "features.csv"), index=False)
        pd.Series(prices, name="price").to_csv(os.path.join(args.out, "prices.csv"), index=False)

    print("\nSaved:")
    print(" -", f_path)
    print(" -", p_path)
    print(" -", d_path)
    if args.save_csv:
        print(" -", os.path.join(args.out, "features.csv"))
        print(" -", os.path.join(args.out, "prices.csv"))


if __name__ == "__main__":
    main()
