from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import logging


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.logging_config import setup_logging
from core.config import load_config
from features.indicators import build_indicators
from data.finnhub_api import AlphaVantageNewsClient

# ---------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------
def rolling_zscore(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling z-score scaling using only past data.
    min_periods=window means intentionally produce NaNs at the beginning,
    then I drop them after the full feature set is built.
    """
    mu = x.rolling(window=window, min_periods=window).mean()
    sigma = x.rolling(window=window, min_periods=window).std(ddof=0)
    # avoid division by zero
    sigma = sigma.replace(0.0, np.nan)  
    return (x - mu) / sigma


# ---------------------------------------------------------------------
# Market data download
# ---------------------------------------------------------------------
def download_ohlcv(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str,
) -> pd.DataFrame:
    """
    Download OHLCV for multiple tickers via yfinance.
    The output columns are MultiIndex: (field, ticker)
    """
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    # Safety check
    if raw is None or raw.empty:
        raise RuntimeError("No data returned from yfinance. Check tickers/date range/interval.")

    raw = raw.sort_index()

    # yfinance can be messy sometimes
    raw = raw[~raw.index.duplicated(keep="first")]  
    return raw


# ---------------------------------------------------------------------
# AlphaVantage news → daily sentiment
# ---------------------------------------------------------------------
def news_feed_to_daily_sentiment(feed_items: list, ticker: str) -> pd.DataFrame:
    """
    Convert AlphaVantage 'news feed' to a daily sentiment series for one ticker.

    Output columns:
      - date (datetime64[ns], normalized at midnight UTC)
      - sentiment: weighted avg of ticker_sentiment_score weighted by relevance_score
      - sentiment_mass: sum(relevance_score) as a 'indicator' for news volume / importance
    """
    if not feed_items:
        return pd.DataFrame(columns=["date", "sentiment", "sentiment_mass"])

    rows = []
    for item in feed_items:
        time_publised = item.get("time_published")
        if not time_publised:
            continue

        # Example: 20221025T113100 (UTC)
        news_date = pd.to_datetime(time_publised, format="%Y%m%dT%H%M%S", utc=True).normalize()

        # Find this ticker entry inside the per-ticker sentiment list
        ts_list = item.get("ticker_sentiment", []) or []
        ts = next((x for x in ts_list if x.get("ticker") == ticker), None)
        if ts is None:
            continue

        try:
            rel = float(ts.get("relevance_score", 0.0) or 0.0)
            sent = float(ts.get("ticker_sentiment_score", 0.0) or 0.0)
        except Exception:
            rel, sent = 0.0, 0.0

        rows.append((news_date, rel, sent))

    if not rows:
        return pd.DataFrame(columns=["date", "sentiment", "sentiment_mass"])

    df = pd.DataFrame(rows, columns=["date", "rel", "sent"])
    df["sent_x_rel"] = df["sent"] * df["rel"]

    daily = (
        df.groupby("date", as_index=False)
        .agg(sentiment_mass=("rel", "sum"), sent_rel=("sent_x_rel", "sum"))
    )

    daily["sentiment"] = np.where(
        daily["sentiment_mass"] > 0,
        daily["sent_rel"] / daily["sentiment_mass"],
        0.0,
    )

    daily = daily[["date", "sentiment", "sentiment_mass"]]
    return daily


def align_daily_sentiment(feature_index: pd.DatetimeIndex, daily: pd.DataFrame) -> pd.DataFrame:
    # Align daily sentiment to the feature index.
    index = pd.to_datetime(feature_index, utc=True).normalize()

    out = pd.DataFrame(index=index, data={"sentiment": 0.0, "sentiment_mass": 0.0})

    if daily is not None and not daily.empty:
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"], utc=True).dt.normalize()
        daily = daily.set_index("date")[["sentiment", "sentiment_mass"]]
        out = out.join(daily, how="left", rsuffix="_src")
        
        # keep defaults where missing
        out["sentiment"] = out["sentiment_src"].fillna(out["sentiment"])
        out["sentiment_mass"] = out["sentiment_mass_src"].fillna(out["sentiment_mass"])
        out = out.drop(columns=["sentiment_src", "sentiment_mass_src"])

    # restore original index shape (timestamps), but keep aligned values
    out.index = feature_index
    return out.astype(np.float32)


# ---------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------
def build_features_for_all_tickers(cfg, raw: pd.DataFrame, logger) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Build a single stacked dataset across tickers.

    Returns:
      - features: (N, D) float32
      - prices:   (N,) float32 (close prices aligned to feature rows)
      - debug_df: per-ticker debug summary
      - meta_df:  per-row metadata (ticker, date)
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError(
            "Expected MultiIndex columns from yfinance (multiple tickers). "
            "If there is only 1 ticker, yfinance can return flat columns."
        )
    
    # list of indicators in the config.yaml
    indicators = list(cfg.features.technical_indicators)

    scaling_cfg = cfg.features.scaling or {}
    scaling_method = scaling_cfg.get("method")
    scaling_window = int(scaling_cfg.get("window") or 0)

    # Flag to use sentiment (via config.yaml)
    use_sentiment = bool(getattr(cfg.features, "include_sentiment", True))

    # Array 
    all_features = []
    all_prices = []
    all_meta = []
    debug_rows = []

    for ticker in cfg.data.tickers:
        # Slice per-ticker: columns become [Open, High, Low, Close, Adj Close, Volume]
        df_t = raw.xs(ticker, axis=1, level=1, drop_level=True).copy()
        df_t.columns = [str(c).strip().lower() for c in df_t.columns]
        df_t = df_t.sort_index()

        # Prefer Adj Close for price alignment (dividends/splits) because I noticed that sometimes close is not available
        if "adj close" in df_t.columns:
            close = df_t["adj close"].astype(float)
        elif "close" in df_t.columns:
            close = df_t["close"].astype(float)
        else:
            raise ValueError(f"{ticker}: missing 'adj close'/'close'. columns={list(df_t.columns)}")

        open_ = df_t["open"].astype(float)
        high = df_t["high"].astype(float)
        low = df_t["low"].astype(float)
        volume = df_t["volume"].astype(float)

        # print(f"\n=== {ticker} ===")
        logger.info(f"\n=== {ticker} ===")
        logger.info(f"[{ticker}] ohlcv rows={len(df_t)} close_non_null={int(close.notna().sum())}")

        feats = build_indicators(
            close=close,
            open_=open_,
            high=high,
            low=low,
            volume=volume,
            indicators=indicators,
        )

        # Debug
        logger.info(f"[{ticker}] feats_raw rows={len(feats)} dim={feats.shape[1]}")
        logger.info(f"[{ticker}] top NaNs:\n{feats.isna().sum().sort_values(ascending=False).head(10)}")

        # Sentiment feature
        if use_sentiment:
            # Currently using hardcode dates because AlphaVantage free plan has date limitations
            ######################################
            date_from = "2018-01-01"
            date_to = "2024-01-01"

            logger.info(f"[{ticker}] fetching AlphaVantage news {date_from} → {date_to}")
            try:
                api_key = str(cfg.alphavantage.api_key)
                av = AlphaVantageNewsClient(api_key=api_key)
                feed = av.fetch_news(
                    tickers=ticker,
                    date_from=date_from,
                    date_to=date_to,
                    limit=1000,
                    use_cache=True,
                )

                daily = news_feed_to_daily_sentiment(feed, ticker=ticker)
                sent_df = align_daily_sentiment(feats.index, daily)

                feats["sentiment"] = sent_df["sentiment"]
                feats["sentiment_mass"] = sent_df["sentiment_mass"]

                logger.info(
                    f"[{ticker}] sentiment min/max={feats['sentiment'].min():.3f}/{feats['sentiment'].max():.3f} "
                    f"mass_sum={feats['sentiment_mass'].sum():.2f}"
                )
            except Exception as e:
                # Sentiment is optional: if it fails I provide default values.
                feats["sentiment"] = 0.0
                feats["sentiment_mass"] = 0.0
                logger.warning(f"[WARN] [{ticker}] AlphaVantage sentiment failed: {e}")
        else:
            feats["sentiment"] = 0.0
            feats["sentiment_mass"] = 0.0

        sent_cols = ["sentiment", "sentiment_mass"]
        tech_cols = [c for c in feats.columns if c not in sent_cols]

        if scaling_method:
            if scaling_method == "rolling_zscore":
                if scaling_window <= 1:
                    raise ValueError("rolling_zscore requires scaling.window > 1")

                feats[tech_cols] = rolling_zscore(feats[tech_cols], scaling_window)
                
                # safe clipping
                feats[tech_cols] = feats[tech_cols].clip(-10, 10)

                # sentiment is already bounded-ish; keeping it stable
                feats["sentiment"] = feats["sentiment"].clip(-1, 1)
                feats["sentiment_mass"] = feats["sentiment_mass"].clip(0, 10)

        logger.info("[{ticker}] rows before dropna={len(feats)}")
        feats = feats.dropna()
        logger.info("[{ticker}] rows after dropna={len(feats)}")

        # Align prices to the final feature rows
        px = close.loc[feats.index].astype(float)

        # Per-row metadata (used later by NLP layer to map row→ticker/date)
        meta_t = pd.DataFrame(
            {"ticker": ticker, "date": feats.index.astype("datetime64[ns]")}
        )
        all_meta.append(meta_t)

        all_features.append(feats)
        all_prices.append(px)

        debug_rows.append(
            {
                "ticker": ticker,
                "rows_raw": int(len(df_t)),
                "rows_features": int(len(feats)),
                "feature_dim": int(feats.shape[1]),
                "first_date": str(feats.index.min()),
                "last_date": str(feats.index.max()),
            }
        )

    features_df = pd.concat(all_features, axis=0)
    prices_s = pd.concat(all_prices, axis=0)
    meta_df = pd.concat(all_meta, axis=0).reset_index(drop=True)

    if len(features_df) != len(prices_s) or len(features_df) != len(meta_df):
        raise RuntimeError(
            f"Alignment mismatch: features={len(features_df)} prices={len(prices_s)} meta={len(meta_df)}"
        )

    return (
        features_df.values.astype("float32"),
        prices_s.values.astype("float32"),
        pd.DataFrame(debug_rows),
        meta_df,
    )


# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------
def main():
    # Enabling logging
    setup_logging("logs/build_feature.log", level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting feature build...")

    # Parsing parameters
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="data")
    p.add_argument("--save_csv", action="store_true", help="Also save features.csv/prices.csv for debugging")
    args = p.parse_args()

    # Loading configuration 
    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    tickers = list(cfg.data.tickers)
    start_date = str(cfg.data.start_date)
    end_date = str(cfg.data.end_date)
    interval = str(cfg.data.interval)

    logger.debug("=== BUILD FEATURES ===")
    logger.debug("config: ", args.config)
    logger.debug("out:", args.out)
    logger.debug("tickers:", tickers)
    logger.debug("range:", start_date, "→", end_date, "interval:", interval)
    logger.debug("indicators:", list(cfg.features.technical_indicators))
    logger.debug("scaling:", cfg.features.scaling)
    logger.debug("include_sentiment:", getattr(cfg.features, "include_sentiment", True))

    raw = download_ohlcv(tickers, start_date, end_date, interval)
    logger.debug(f"\nDownloaded raw: shape={raw.shape}")
    logger.debug("raw index head:", raw.index[:3])
    logger.debug("raw index tail:", raw.index[-3:])
    logger.debug("raw columns sample:", list(raw.columns)[:10])

    features, prices, dbg, meta_df = build_features_for_all_tickers(cfg, raw, logger)

    # -----------------------
    # Validation checks
    # -----------------------
    logger.debug("\n=== VALIDATION ===")
    logger.debug(dbg.to_string(index=False))
    logger.debug("features shape:", features.shape, "dtype:", features.dtype)
    logger.debug("prices shape:", prices.shape, "dtype:", prices.dtype)
    logger.debug("meta_df shape:", meta_df.shape)

    if len(features) != len(prices) or len(features) != len(meta_df):
        raise RuntimeError(
            f"Alignment mismatch: features={len(features)} prices={len(prices)} meta={len(meta_df)}"
        )

    # Safety check for both features and prices
    if not np.isfinite(features).all():
        raise RuntimeError("features contains NaN/Inf after dropna. Check indicators/scaling window.")
    if not np.isfinite(prices).all():
        raise RuntimeError("prices contains NaN/Inf - unexpected.")

    for col in ("ticker", "date"):
        if col not in meta_df.columns:
            raise RuntimeError(f"meta_df missing '{col}'. columns={list(meta_df.columns)}")

    meta_df = meta_df.copy()
    meta_df["date"] = pd.to_datetime(meta_df["date"], errors="coerce")
    if meta_df["date"].isna().any():
        bad = meta_df[meta_df["date"].isna()].head(5)
        raise RuntimeError(f"meta_df has invalid dates. examples:\n{bad}")

    unknown = set(meta_df["ticker"].unique()) - set(tickers)
    if unknown:
        raise RuntimeError(f"meta_df contains unexpected tickers: {sorted(list(unknown))}")

    meta_summary = (
        meta_df.groupby("ticker")["date"]
        .agg(rows="size", first="min", last="max")
        .reset_index()
    )

    logger.debug("\n=== META SUMMARY ===")
    logger.debug(meta_summary.to_string(index=False))

    # -----------------------
    # Save features, prices and meta data (parquet format)
    # -----------------------
    f_path = os.path.join(args.out, "features.npy")
    p_path = os.path.join(args.out, "prices.npy")
    d_path = os.path.join(args.out, "build_debug.csv")

    m_parquet = os.path.join(args.out, "row_meta.parquet")
    m_summary = os.path.join(args.out, "row_meta_summary.csv")

    np.save(f_path, features)
    np.save(p_path, prices)
    dbg.to_csv(d_path, index=False)

    meta_df.to_parquet(m_parquet, index=False)
    meta_summary.to_csv(m_summary, index=False)

    if args.save_csv:
        pd.DataFrame(features).to_csv(os.path.join(args.out, "features.csv"), index=False)
        pd.Series(prices, name="price").to_csv(os.path.join(args.out, "prices.csv"), index=False)
        meta_df.to_csv(os.path.join(args.out, "row_meta.csv"), index=False)

    logger.info("\nSaved:")
    logger.info(" -%s", f_path)
    logger.info(" -%s", p_path)
    logger.info(" -%s", d_path)
    logger.info(" -%s", m_parquet)
    logger.info(" -%s", m_summary)
    if args.save_csv:
        logger.info(" -%s", os.path.join(args.out, "features.csv"))
        logger.info(" -%s", os.path.join(args.out, "prices.csv"))
        logger.info(" -%s", os.path.join(args.out, "row_meta.csv"))


if __name__ == "__main__":
    main()
