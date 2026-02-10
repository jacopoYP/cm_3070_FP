# from __future__ import annotations

# import argparse
# import os
# import sys
# from dataclasses import asdict
# from typing import Tuple

# import numpy as np
# import pandas as pd
# import yfinance as yf

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from core.config import load_config
# from features.indicators import build_indicators

# from data.finnhub_api import AlphaVantageNewsClient

# def _rolling_zscore(x: pd.DataFrame, window: int) -> pd.DataFrame:
#     """
#     Rolling z-score scaling. Uses only past data (no look-ahead).
#     """
#     mu = x.rolling(window=window, min_periods=window).mean()
#     sigma = x.rolling(window=window, min_periods=window).std(ddof=0)
#     return (x - mu) / (sigma.replace(0.0, np.nan))


# def download_ohlcv(tickers, start_date, end_date, interval) -> pd.DataFrame:
#     df = yf.download(
#         tickers=tickers,
#         start=start_date,
#         end=end_date,
#         interval=interval,
#         auto_adjust=False,
#         progress=False,
#         group_by="column",   # MultiIndex columns: (field, ticker)
#         threads=True,
#     )
#     if df is None or df.empty:
#         raise RuntimeError("No data returned. Check tickers/date range.")
#     return df

# def _to_daily_sentiment_alpha(feed_items: list, ticker: str) -> pd.DataFrame:
#     """
#     feed_items: list of AlphaVantage 'feed' dicts
#     returns: df with columns [date, sentiment, sentiment_mass]
#     sentiment = weighted avg of ticker_sentiment_score weighted by relevance_score
#     sentiment_mass = sum(relevance_score) (proxy for volume/importance)
#     """
#     if not feed_items:
#         return pd.DataFrame(columns=["date", "sentiment", "sentiment_mass"])

#     rows = []
#     for item in feed_items:
#         tp = item.get("time_published")
#         if not tp:
#             continue

#         # UTC timestamp e.g. 20221025T113100
#         dt = pd.to_datetime(tp, format="%Y%m%dT%H%M%S", utc=True)
#         d = dt.date()

#         # find this ticker in per-ticker sentiment list
#         ts_list = item.get("ticker_sentiment", []) or []
#         for ts in ts_list:
#             if ts.get("ticker") == ticker:
#                 try:
#                     rel = float(ts.get("relevance_score", "0") or 0.0)
#                     sent = float(ts.get("ticker_sentiment_score", "0") or 0.0)
#                 except Exception:
#                     rel, sent = 0.0, 0.0
#                 rows.append((d, rel, sent))
#                 break

#     if not rows:
#         return pd.DataFrame(columns=["date", "sentiment", "sentiment_mass"])

#     df = pd.DataFrame(rows, columns=["date", "rel", "sent"])

#     g = df.groupby("date", as_index=False).agg(
#         rel_sum=("rel", "sum"),
#         sent_rel_sum=("sent", lambda s: 0.0),  # placeholder, we’ll compute below
#     )

#     # compute sum(sent*rel) per date
#     sent_rel = df.assign(sent_rel=df["sent"] * df["rel"]).groupby("date", as_index=False)["sent_rel"].sum()
#     g = g.drop(columns=["sent_rel_sum"]).merge(sent_rel, on="date", how="left")

#     g["sentiment"] = g["sent_rel"] / (g["rel_sum"] + 1e-9)
#     g = g.rename(columns={"rel_sum": "sentiment_mass"})
#     g = g[["date", "sentiment", "sentiment_mass"]]
#     return g

# def _align_daily_sentiment_to_index(feat_index: pd.DatetimeIndex, daily: pd.DataFrame) -> pd.DataFrame:
#     idx_dates = pd.to_datetime(feat_index).date

#     out = pd.DataFrame(index=idx_dates, data={
#         "sentiment": 0.0,
#         "sentiment_mass": 0.0
#     })

#     if daily is not None and not daily.empty:
#         for _, row in daily.iterrows():
#             d = row["date"]
#             if d in out.index:   # <-- IMPORTANT: prevent index expansion
#                 out.loc[d, "sentiment"] = float(row["sentiment"])
#                 out.loc[d, "sentiment_mass"] = float(row["sentiment_mass"])

#     out.index = feat_index
#     return out.astype(np.float32)


# def build_features_for_all_tickers(cfg, raw: pd.DataFrame):
#     if not isinstance(raw.columns, pd.MultiIndex):
#         raise ValueError("Expected MultiIndex columns from yfinance when downloading multiple tickers.")

#     indicators = list(cfg.features.technical_indicators)

#     scaling_cfg = cfg.features.scaling or {}
#     scaling_method = scaling_cfg.get("method")
#     scaling_window = scaling_cfg.get("window")

#     all_features = []
#     all_prices = []
#     debug_rows = []
#     all_meta = []

#     for t in cfg.data.tickers:
#         # Slice one ticker from MultiIndex columns: (field, ticker)
#         df_t = raw.xs(t, axis=1, level=1, drop_level=True)  # columns become ['Adj Close','Close','High','Low','Open','Volume']

#         # Standardize column names to lower
#         df_t.columns = [str(c).strip().lower() for c in df_t.columns]

#         # Pick OHLCV (Adj Close preferred)
#         if "adj close" in df_t.columns:
#             close = df_t["adj close"].astype(float)
#         elif "close" in df_t.columns:
#             close = df_t["close"].astype(float)
#         else:
#             raise ValueError(f"{t}: missing 'adj close'/'close' in columns {list(df_t.columns)}")

#         open_ = df_t["open"].astype(float)
#         high  = df_t["high"].astype(float)
#         low   = df_t["low"].astype(float)
#         vol   = df_t["volume"].astype(float)

#         print(f"\n=== {t} ===")
#         print("df_t rows:", len(df_t))
#         print("close non-null:", close.notna().sum())

#         feats = build_indicators(
#             close=close,
#             open_=open_,
#             high=high,
#             low=low,
#             volume=vol,
#             indicators=indicators,
#         )

#         print("feats raw rows:", len(feats))
#         print("feats NaNs per column:\n", feats.isna().sum().sort_values(ascending=False).head(10))

#         # Sentiment
#                 # -----------------------
#         # Sentiment feature (v0)
#         # -----------------------
#         # use_sent = bool(getattr(getattr(cfg, "sentiment", {}), "enabled", True))
#         # if use_sent:
#             # You can keep it simple: fetch sentiment only for the feature range
#             # But Finnhub free tier may only provide recent history, so fallback to 0.0 is OK.
#         # date_from = str(feats.index.min().date())
#         # date_to   = str(feats.index.max().date())
#         date_from = "2018-01-01"
#         date_to = "2024-01-01"

#         print("Start fetching news")

#         print("Start fetching AlphaVantage news")

#         try:
#             av = AlphaVantageNewsClient(api_key=str(cfg.alphavantage.api_key))
#             feed = av.fetch_news(tickers=t, date_from=date_from, date_to=date_to, limit=1000, use_cache=True)
#             print("n_feed:", len(feed))

#             daily = _to_daily_sentiment_alpha(feed, ticker=t)
#             sent_df = _align_daily_sentiment_to_index(feats.index, daily)

#             feats["sentiment"] = sent_df["sentiment"]
#             feats["sentiment_mass"] = sent_df["sentiment_mass"]

#         except Exception as e:
#             print("Error fetching AlphaVantage news")
#             feats["sentiment"] = 0.0
#             feats["sentiment_mass"] = 0.0
#             print(f"[WARN] AlphaVantage sentiment failed for {t}: {e}")

#         print("sentiment stats:", feats["sentiment"].min(), feats["sentiment"].max(), "mass sum:", feats["sentiment_mass"].sum())

#         sent_cols = [c for c in ["sentiment", "sentiment_mass"] if c in feats.columns]
#         tech_cols = [c for c in feats.columns if c not in sent_cols]

#         if scaling_method is not None:
#             if scaling_method == "rolling_zscore":
#                 feats[tech_cols] = _rolling_zscore(feats[tech_cols], int(scaling_window))
#                 feats[tech_cols] = feats[tech_cols].clip(-10, 10)  # optional, recommended

#                 # keep sentiment as-is, just clip to safe ranges
#                 if "sentiment" in feats.columns:
#                     feats["sentiment"] = feats["sentiment"].clip(-1, 1)
#                 if "sentiment_mass" in feats.columns:
#                     feats["sentiment_mass"] = feats["sentiment_mass"].clip(0, 10)

#         # BEFORE dropna
#         print("rows before dropna:", len(feats))

#         feats = feats.dropna()
#         print("rows after dropna:", len(feats))

#         px = close.loc[feats.index].astype(float)

#         # all_meta contains info to be queried later by the use via NLP 
#         meta_t = pd.DataFrame({
#             "ticker": t,
#             "date": feats.index.astype("datetime64[ns]"),
#         })
#         all_meta.append(meta_t)

#         all_features.append(feats)
#         all_prices.append(px)

#         debug_rows.append({
#             "ticker": t,
#             "rows_raw": int(len(df_t)),
#             "rows_features": int(len(feats)),
#             "feature_dim": int(feats.shape[1]),
#             "first_date": str(feats.index.min()),
#             "last_date": str(feats.index.max()),
#         })

#     features_df = pd.concat(all_features, axis=0)
#     prices_s = pd.concat(all_prices, axis=0)
#     meta_df = pd.concat(all_meta, axis=0).reset_index(drop=True)

#     if len(features_df) != len(prices_s):
#         raise RuntimeError(f"Alignment mismatch: features={len(features_df)} prices={len(prices_s)}")
    
#     if len(features_df) != len(prices_s) or len(features_df) != len(meta_df):
#         raise RuntimeError(
#          f"Alignment mismatch: features={len(features_df)} prices={len(prices_s)} meta={len(meta_df)}"
#         )


#     # return features_df.values.astype("float32"), prices_s.values.astype("float32"), pd.DataFrame(debug_rows)
#     return (
#         features_df.values.astype("float32"),
#         prices_s.values.astype("float32"),
#         pd.DataFrame(debug_rows),
#         meta_df,
#     )

# # Main
# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--config", default="config.yaml")
#     p.add_argument("--out", default="data")
#     p.add_argument("--save_csv", action="store_true", help="Also save features.csv and prices.csv for debugging")
#     args = p.parse_args()

#     cfg = load_config(args.config)

#     # Config loaded by config.yaml
#     data_cfg = getattr(cfg, "data", cfg)

#     tickers = list(getattr(data_cfg, "tickers"))
#     start_date = str(getattr(data_cfg, "start_date"))
#     end_date = str(getattr(data_cfg, "end_date"))
#     interval = str(getattr(data_cfg, "interval"))


#     os.makedirs(args.out, exist_ok=True)

#     print("=== BUILD FEATURES ===")
#     print("config:", args.config)
#     print("out:", args.out)
#     print("tickers:", tickers)
#     print("range:", start_date, "→", end_date, "interval:", interval)
#     print("indicators:", list(cfg.features.technical_indicators))
#     print("scaling:", cfg.features.scaling)

#     print("AlphaVantageConfig API:", cfg.alphavantage.api_key)


#     # Download Open, High, Low, Close and Volume data per ticker
#     raw = download_ohlcv(tickers, start_date, end_date, interval)
#     print("downloaded rows:", len(raw), "cols:", list(raw.columns))
#     print("RAW shape:", raw.shape)
#     print("RAW head index:", raw.index[:3])
#     print("RAW tail index:", raw.index[-3:])
#     print("RAW columns (first 10):", list(raw.columns)[:10])

#     # Building the features data set
#     features, prices, dbg, meta_df = build_features_for_all_tickers(cfg, raw)

#     # Validate
#     print("\n=== VALIDATION ===")
#     print(dbg.to_string(index=False))
#     print("features shape:", features.shape)
#     print("prices shape:", prices.shape)
#     print("meta_df shape:", meta_df.shape)
#     print("features dtype:", features.dtype, "prices dtype:", prices.dtype)

#     # Safety check
#     if len(features) != len(prices) or len(features) != len(meta_df):
#         raise RuntimeError(
#             f"Alignment mismatch: features={len(features)} prices={len(prices)} meta={len(meta_df)}"
#         )

#     # Numeric checks
#     if not np.isfinite(features).all():
#         raise RuntimeError("features contains NaN/Inf after dropna - check scaling window / indicator math.")
#     if not np.isfinite(prices).all():
#         raise RuntimeError("prices contains NaN/Inf - unexpected.")
    
#     # Other safety check for required meta columns
#     for col in ("ticker", "date"):
#         if col not in meta_df.columns:
#             raise RuntimeError(f"meta_df missing required column '{col}'. Columns={list(meta_df.columns)}")
        
#     # Ensure date is datetime64
#     meta_df = meta_df.copy()
#     meta_df["date"] = pd.to_datetime(meta_df["date"], errors="coerce")
#     if meta_df["date"].isna().any():
#         bad = meta_df[meta_df["date"].isna()].head(5)
#         raise RuntimeError(f"meta_df has invalid dates. Examples:\n{bad}")

#     # Ensure tickers are in the allowed universe
#     unknown = set(meta_df["ticker"].unique()) - set(tickers)
#     if unknown:
#         raise RuntimeError(f"meta_df contains unexpected tickers: {sorted(list(unknown))}")

#     # Quick summary (useful for debugging)
#     meta_summary = (
#         meta_df.groupby("ticker")["date"]
#         .agg(rows="size", first="min", last="max")
#         .reset_index()
#     )
#     print("\n=== META SUMMARY ===")
#     print(meta_summary.to_string(index=False))

#     # Saving artifacts
#     f_path = os.path.join(args.out, "features.npy")
#     p_path = os.path.join(args.out, "prices.npy")
#     d_path = os.path.join(args.out, "build_debug.csv")

#     # Meta artifacts
#     m_parquet = os.path.join(args.out, "row_meta.parquet")
#     m_summary = os.path.join(args.out, "row_meta_summary.csv")

#     np.save(f_path, features)
#     np.save(p_path, prices)
#     dbg.to_csv(d_path, index=False)

#     # meta saving
#     meta_df.to_parquet(m_parquet, index=False)
#     meta_summary.to_csv(m_summary, index=False)

#     if args.save_csv:
#         # Used only for debugging
#         pd.DataFrame(features).to_csv(os.path.join(args.out, "features.csv"), index=False)
#         pd.Series(prices, name="price").to_csv(os.path.join(args.out, "prices.csv"), index=False)
#         meta_df.to_csv(os.path.join(args.out, "row_meta.csv"), index=False)

#     print("\nSaved:")
#     print(" -", f_path)
#     print(" -", p_path)
#     print(" -", d_path)
#     print(" -", m_parquet)
#     print(" -", m_summary)
#     if args.save_csv:
#         print(" -", os.path.join(args.out, "features.csv"))
#         print(" -", os.path.join(args.out, "prices.csv"))
#         print(" -", os.path.join(args.out, "row_meta.csv"))

# if __name__ == "__main__":
#     main()


from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
        tp = item.get("time_published")
        if not tp:
            continue

        # Example: 20221025T113100 (UTC)
        dt = pd.to_datetime(tp, format="%Y%m%dT%H%M%S", utc=True).normalize()

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

        rows.append((dt, rel, sent))

    if not rows:
        return pd.DataFrame(columns=["date", "sentiment", "sentiment_mass"])

    df = pd.DataFrame(rows, columns=["date", "rel", "sent"])
    df["sent_x_rel"] = df["sent"] * df["rel"]

    daily = (
        df.groupby("date", as_index=False)
        .agg(sentiment_mass=("rel", "sum"), sent_rel=("sent_x_rel", "sum"))
    )

    # daily["sentiment"] = daily["sent_rel"] / (daily["sentiment_mass"] + 1e-9)
    daily["sentiment"] = np.where(
        daily["sentiment_mass"] > 0,
        daily["sent_rel"] / daily["sentiment_mass"],
        0.0,
    )

    daily = daily[["date", "sentiment", "sentiment_mass"]]
    return daily


def align_daily_sentiment(feature_index: pd.DatetimeIndex, daily: pd.DataFrame) -> pd.DataFrame:
    """
    Align daily sentiment to the feature index.

    Strategy:
    - normalize feature timestamps to daily buckets
    - left-join daily sentiment
    - fill missing days with 0.0 (neutral / no news)
    """
    idx = pd.to_datetime(feature_index, utc=True).normalize()

    out = pd.DataFrame(index=idx, data={"sentiment": 0.0, "sentiment_mass": 0.0})

    if daily is not None and not daily.empty:
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"], utc=True).dt.normalize()
        daily = daily.set_index("date")[["sentiment", "sentiment_mass"]]
        out = out.join(daily, how="left", rsuffix="_src")
        
        # keep defaults where missing
        out["sentiment"] = out["sentiment_src"].fillna(out["sentiment"])
        out["sentiment_mass"] = out["sentiment_mass_src"].fillna(out["sentiment_mass"])
        out = out.drop(columns=["sentiment_src", "sentiment_mass_src"])

    # restore original index shape (timestamps) but keep aligned values
    out.index = feature_index
    return out.astype(np.float32)


# ---------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------
def build_features_for_all_tickers(cfg, raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
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

        # Prefer Adj Close for price alignment (dividends/splits)
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

        print(f"\n=== {ticker} ===")
        print(f"[{ticker}] ohlcv rows={len(df_t)} close_non_null={int(close.notna().sum())}")

        feats = build_indicators(
            close=close,
            open_=open_,
            high=high,
            low=low,
            volume=volume,
            indicators=indicators,
        )

        print(f"[{ticker}] feats_raw rows={len(feats)} dim={feats.shape[1]}")
        print(f"[{ticker}] top NaNs:\n{feats.isna().sum().sort_values(ascending=False).head(10)}")

        # Sentiment feature
        if use_sentiment:
            

            ######################################
            # Currently using hardcode dates because AlphaVantage free plan has date limitations
            # Keeping the code commented for future implementations
            ######################################
            # date_from = str(feats.index.min().date())
            # date_to = str(feats.index.max().date())
            date_from = "2018-01-01"
            date_to = "2024-01-01"

            print(f"[{ticker}] fetching AlphaVantage news {date_from} → {date_to}")
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

                print(
                    f"[{ticker}] sentiment min/max={feats['sentiment'].min():.3f}/{feats['sentiment'].max():.3f} "
                    f"mass_sum={feats['sentiment_mass'].sum():.2f}"
                )
            except Exception as e:
                # Sentiment is an optional feature: if it fails the pipeline should run anyway.
                feats["sentiment"] = 0.0
                feats["sentiment_mass"] = 0.0
                print(f"[WARN] [{ticker}] AlphaVantage sentiment failed: {e}")
        else:
            feats["sentiment"] = 0.0
            feats["sentiment_mass"] = 0.0

        # Scaling
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

        print(f"[{ticker}] rows before dropna={len(feats)}")
        feats = feats.dropna()
        print(f"[{ticker}] rows after dropna={len(feats)}")

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
# Script Entrypoint
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="data")
    p.add_argument("--save_csv", action="store_true", help="Also save features.csv/prices.csv for debugging")
    args = p.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    tickers = list(cfg.data.tickers)
    start_date = str(cfg.data.start_date)
    end_date = str(cfg.data.end_date)
    interval = str(cfg.data.interval)

    print("=== BUILD FEATURES ===")
    print("config:", args.config)
    print("out:", args.out)
    print("tickers:", tickers)
    print("range:", start_date, "→", end_date, "interval:", interval)
    print("indicators:", list(cfg.features.technical_indicators))
    print("scaling:", cfg.features.scaling)
    print("include_sentiment:", getattr(cfg.features, "include_sentiment", True))

    raw = download_ohlcv(tickers, start_date, end_date, interval)
    print(f"\nDownloaded raw: shape={raw.shape}")
    print("raw index head:", raw.index[:3])
    print("raw index tail:", raw.index[-3:])
    print("raw columns sample:", list(raw.columns)[:10])

    features, prices, dbg, meta_df = build_features_for_all_tickers(cfg, raw)

    # -----------------------
    # Validation checks
    # -----------------------
    print("\n=== VALIDATION ===")
    print(dbg.to_string(index=False))
    print("features shape:", features.shape, "dtype:", features.dtype)
    print("prices shape:", prices.shape, "dtype:", prices.dtype)
    print("meta_df shape:", meta_df.shape)

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

    print("\n=== META SUMMARY ===")
    print(meta_summary.to_string(index=False))

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

    print("\nSaved:")
    print(" -", f_path)
    print(" -", p_path)
    print(" -", d_path)
    print(" -", m_parquet)
    print(" -", m_summary)
    if args.save_csv:
        print(" -", os.path.join(args.out, "features.csv"))
        print(" -", os.path.join(args.out, "prices.csv"))
        print(" -", os.path.join(args.out, "row_meta.csv"))


if __name__ == "__main__":
    main()
