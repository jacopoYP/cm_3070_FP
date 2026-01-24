from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import load_config
from features.indicators import build_indicators

from data.finnhub_api import AlphaVantageNewsClient

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

def _to_daily_sentiment_alpha(feed_items: list, ticker: str) -> pd.DataFrame:
    """
    feed_items: list of AlphaVantage 'feed' dicts
    returns: df with columns [date, sentiment, sentiment_mass]
    sentiment = weighted avg of ticker_sentiment_score weighted by relevance_score
    sentiment_mass = sum(relevance_score) (proxy for volume/importance)
    """
    if not feed_items:
        return pd.DataFrame(columns=["date", "sentiment", "sentiment_mass"])

    rows = []
    for item in feed_items:
        tp = item.get("time_published")
        if not tp:
            continue

        # UTC timestamp e.g. 20221025T113100
        dt = pd.to_datetime(tp, format="%Y%m%dT%H%M%S", utc=True)
        d = dt.date()

        # find this ticker in per-ticker sentiment list
        ts_list = item.get("ticker_sentiment", []) or []
        for ts in ts_list:
            if ts.get("ticker") == ticker:
                try:
                    rel = float(ts.get("relevance_score", "0") or 0.0)
                    sent = float(ts.get("ticker_sentiment_score", "0") or 0.0)
                except Exception:
                    rel, sent = 0.0, 0.0
                rows.append((d, rel, sent))
                break

    if not rows:
        return pd.DataFrame(columns=["date", "sentiment", "sentiment_mass"])

    df = pd.DataFrame(rows, columns=["date", "rel", "sent"])

    g = df.groupby("date", as_index=False).agg(
        rel_sum=("rel", "sum"),
        sent_rel_sum=("sent", lambda s: 0.0),  # placeholder, we’ll compute below
    )

    # compute sum(sent*rel) per date
    sent_rel = df.assign(sent_rel=df["sent"] * df["rel"]).groupby("date", as_index=False)["sent_rel"].sum()
    g = g.drop(columns=["sent_rel_sum"]).merge(sent_rel, on="date", how="left")

    g["sentiment"] = g["sent_rel"] / (g["rel_sum"] + 1e-9)
    g = g.rename(columns={"rel_sum": "sentiment_mass"})
    g = g[["date", "sentiment", "sentiment_mass"]]
    return g


def _to_daily_sentiment(news_items) -> pd.DataFrame:
    """
    news_items: list of finnhub company_news dicts
    returns: df with columns [date, sentiment, n_articles]
    NOTE: assumes each item already has a per-article sentiment score.
          (If you compute sentiment elsewhere, adapt accordingly.)
    """
    if not news_items:
        return pd.DataFrame(columns=["date", "sentiment", "n_articles"])

    df = pd.DataFrame(news_items)
    if df.empty or "datetime" not in df.columns:
        return pd.DataFrame(columns=["date", "sentiment", "n_articles"])

    # Finnhub datetime is unix seconds
    df["date"] = pd.to_datetime(df["datetime"], unit="s", utc=True).dt.date

    # You must have computed per-article numeric sentiment in a field
    # e.g. item["sentiment"] in [-1..+1] or probability-weighted score
    if "sentiment" not in df.columns:
        raise RuntimeError("News items missing 'sentiment' field. Compute FinBERT score first.")

    out = (
        df.groupby("date")
          .agg(sentiment=("sentiment", "mean"), n_articles=("sentiment", "size"))
          .reset_index()
    )
    return out


def _align_daily_sentiment_to_index(feat_index: pd.DatetimeIndex, daily: pd.DataFrame) -> pd.DataFrame:
    idx_dates = pd.to_datetime(feat_index).date

    out = pd.DataFrame(index=idx_dates, data={
        "sentiment": 0.0,
        "sentiment_mass": 0.0
    })

    if daily is not None and not daily.empty:
        for _, row in daily.iterrows():
            d = row["date"]
            if d in out.index:   # <-- IMPORTANT: prevent index expansion
                out.loc[d, "sentiment"] = float(row["sentiment"])
                out.loc[d, "sentiment_mass"] = float(row["sentiment_mass"])

    out.index = feat_index
    return out.astype(np.float32)


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

        print(f"\n=== {t} ===")
        print("df_t rows:", len(df_t))
        print("close non-null:", close.notna().sum())

        feats = build_indicators(
            close=close,
            open_=open_,
            high=high,
            low=low,
            volume=vol,
            indicators=indicators,
        )

        print("feats raw rows:", len(feats))
        print("feats NaNs per column:\n", feats.isna().sum().sort_values(ascending=False).head(10))

       
        # Sentiment
                # -----------------------
        # Sentiment feature (v0)
        # -----------------------
        # use_sent = bool(getattr(getattr(cfg, "sentiment", {}), "enabled", True))
        # if use_sent:
            # You can keep it simple: fetch sentiment only for the feature range
            # But Finnhub free tier may only provide recent history, so fallback to 0.0 is OK.
        # date_from = str(feats.index.min().date())
        # date_to   = str(feats.index.max().date())
        date_from = "2018-01-01"
        date_to = "2024-01-01"

        print("Start fetching news")

        print("Start fetching AlphaVantage news")

        try:
            av = AlphaVantageNewsClient(api_key=str(cfg.alphavantage.api_key))
            feed = av.fetch_news(tickers=t, date_from=date_from, date_to=date_to, limit=1000, use_cache=True)
            print("n_feed:", len(feed))

            daily = _to_daily_sentiment_alpha(feed, ticker=t)
            sent_df = _align_daily_sentiment_to_index(feats.index, daily)

            feats["sentiment"] = sent_df["sentiment"]
            feats["sentiment_mass"] = sent_df["sentiment_mass"]

        except Exception as e:
            print("Error fetching AlphaVantage news")
            feats["sentiment"] = 0.0
            feats["sentiment_mass"] = 0.0
            print(f"[WARN] AlphaVantage sentiment failed for {t}: {e}")

        print("sentiment stats:", feats["sentiment"].min(), feats["sentiment"].max(), "mass sum:", feats["sentiment_mass"].sum())

        # # # Optional scaling
        # if scaling_method is not None:
        #     if scaling_method == "rolling_zscore":
        #         if scaling_window is None:
        #             raise ValueError("scaling.window must be set for rolling_zscore")
        #         feats = _rolling_zscore(feats, int(scaling_window))
        #     else:
        #         raise ValueError(f"Unsupported scaling.method: {scaling_method}")
        sent_cols = [c for c in ["sentiment", "sentiment_mass"] if c in feats.columns]
        tech_cols = [c for c in feats.columns if c not in sent_cols]

        if scaling_method is not None:
            if scaling_method == "rolling_zscore":
                feats[tech_cols] = _rolling_zscore(feats[tech_cols], int(scaling_window))
                feats[tech_cols] = feats[tech_cols].clip(-10, 10)  # optional, recommended

                # keep sentiment as-is, just clip to safe ranges
                if "sentiment" in feats.columns:
                    feats["sentiment"] = feats["sentiment"].clip(-1, 1)
                if "sentiment_mass" in feats.columns:
                    feats["sentiment_mass"] = feats["sentiment_mass"].clip(0, 10)

        # BEFORE dropna
        print("rows before dropna:", len(feats))

        feats = feats.dropna()
        print("rows after dropna:", len(feats))

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

    print("AlphaVantageConfig API:", cfg.alphavantage.api_key)


    # Step 1: download
    raw = download_ohlcv(tickers, start_date, end_date, interval)
    print("downloaded rows:", len(raw), "cols:", list(raw.columns))
    raw = download_ohlcv(tickers, start_date, end_date, interval)
    print("RAW shape:", raw.shape)
    print("RAW head index:", raw.index[:3])
    print("RAW tail index:", raw.index[-3:])
    print("RAW columns (first 10):", list(raw.columns)[:10])


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
