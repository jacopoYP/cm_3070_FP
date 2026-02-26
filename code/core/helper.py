# sentiment/sentiment_filter.py
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from types import SimpleNamespace
from typing import Tuple
import datetime as dt
import os
import yaml

def check_sentiment(
    x: np.ndarray,
    cfg: Any,
    sentiment_col: int = -2,
    mass_col: int = -1,
) -> bool:
    """
    x: 1D feature vector at time t (shape: [D])
    cfg: config object (expects fields on cfg.trade_manager or cfg directly)
    Sentiment is stored at indices sentiment_col and mass_col.

    Logic:
      - if filter disabled: True
      - if mass < mass_min: True  (no-news => ignore sentiment)
      - else require sentiment >= sentiment_min_score
    """
    # allow cfg to be either cfg.trade_manager or trade_cfg itself
    trade_cfg = getattr(cfg, "trade_manager", cfg)

    if not bool(getattr(trade_cfg, "use_sentiment_filter", False)):
        return True

    mass_min = float(getattr(trade_cfg, "sentiment_mass_min", 0.0))
    sent_min = float(getattr(trade_cfg, "sentiment_min_score", 0.0))

    try:
        sent = float(x[sentiment_col])
        mass = float(x[mass_col])
    except Exception:
        # fail open
        return True

    if mass < mass_min:
        return True

    return sent >= sent_min

def split_by_segments(
    features: np.ndarray,
    prices: np.ndarray,
    train_frac: float,
    args,
    cfg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Stack segments: [seg0, seg1, ...] already concatenated in input arrays.

    For each segment:
      first floor(seg_len * train_frac) bars -> train
      remaining -> test

    Returns:
      X_train, p_train, X_test, p_test, seg_train_len, seg_test_len, n_segs
    """

    meta = pd.read_parquet(args.meta)
    meta["date"] = pd.to_datetime(meta["date"])
    meta["ticker"] = meta["ticker"].astype(str)

    # sanity: must align 1:1 with rows
    if len(meta) != len(prices):
        raise RuntimeError(f"meta mismatch: meta={len(meta)} prices={len(prices)}")

    tickers = list(cfg.data.tickers)
    train_frac = float(getattr(cfg.data, "train_frac", 0.7))

    train_idx = []
    test_idx = []

    for t in tickers:
        idx_t = meta.index[meta["ticker"] == t].to_numpy()
        
        # Ensuring time order
        idx_t = idx_t[np.argsort(meta.loc[idx_t, "date"].to_numpy())]

        cut = int(np.floor(len(idx_t) * train_frac))
        train_idx.append(idx_t[:cut])
        test_idx.append(idx_t[cut:])

    train_idx = np.concatenate(train_idx).astype(np.int64)
    test_idx  = np.concatenate(test_idx).astype(np.int64)

    X_train = features[train_idx]
    p_train = prices[train_idx]
    X_test  = features[test_idx]
    p_test  = prices[test_idx]

    # segment lengths per ticker
    rows_per_ticker = int(meta.groupby("ticker").size().iloc[0])
    seg_train_len = int(np.floor(rows_per_ticker * train_frac))
    seg_test_len = rows_per_ticker - seg_train_len
    n_segs = len(tickers)

    return X_train, p_train, X_test, p_test, seg_train_len, seg_test_len, n_segs

def split_by_ticker_time(
    features: np.ndarray,
    prices: np.ndarray,
    tickers: list[str],
    meta_path: str,
    train_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    meta = pd.read_parquet(meta_path)
    meta["date"] = pd.to_datetime(meta["date"])
    meta["ticker"] = meta["ticker"].astype(str)

    if len(meta) != len(prices):
        raise RuntimeError(f"meta mismatch: meta={len(meta)} prices={len(prices)}")

    sizes = meta.groupby("ticker").size()
    if sizes.nunique() != 1:
        raise RuntimeError(f"Unequal rows per ticker: {sizes.to_dict()}")
    rows_per_ticker = int(sizes.iloc[0])

    train_idx, test_idx = [], []

    for t in tickers:
        idx_t = meta.index[meta["ticker"] == t].to_numpy()
        idx_t = idx_t[np.argsort(meta.loc[idx_t, "date"].to_numpy())]

        cut = int(np.floor(len(idx_t) * train_frac))
        train_idx.append(idx_t[:cut])
        test_idx.append(idx_t[cut:])

    train_idx = np.concatenate(train_idx).astype(np.int64)
    test_idx  = np.concatenate(test_idx).astype(np.int64)

    X_train = features[train_idx]
    p_train = prices[train_idx]
    X_test  = features[test_idx]
    p_test  = prices[test_idx]

    seg_train_len = int(np.floor(rows_per_ticker * train_frac))
    seg_test_len = rows_per_ticker - seg_train_len
    n_segs = len(tickers)

    return X_train, p_train, X_test, p_test, seg_train_len, seg_test_len, n_segs

def split_by_ticker_time_ga(
    features: np.ndarray,
    prices: np.ndarray,
    tickers: List[str],
    meta_path: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[
    np.ndarray, np.ndarray,  # Train
    np.ndarray, np.ndarray,  # Val
    np.ndarray, np.ndarray,  # Test
    int, int, int,           # seg_train_len, seg_val_len, seg_test_len
    int                      # n_segs
]:
    meta = pd.read_parquet(meta_path)
    meta["date"] = pd.to_datetime(meta["date"])
    meta["ticker"] = meta["ticker"].astype(str)

    if len(meta) != len(prices):
        raise RuntimeError(f"meta mismatch: meta={len(meta)} prices={len(prices)}")

    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    sizes = meta.groupby("ticker").size()
    if sizes.nunique() != 1:
        raise RuntimeError(f"Unequal rows per ticker: {sizes.to_dict()}")
    rows_per_ticker = int(sizes.iloc[0])

    train_idx, val_idx, test_idx = [], [], []

    for t in tickers:
        idx_t = meta.index[meta["ticker"] == t].to_numpy()
        idx_t = idx_t[np.argsort(meta.loc[idx_t, "date"].to_numpy())]

        n = len(idx_t)
        cut_train = int(np.floor(n * train_frac))
        cut_val   = int(np.floor(n * (train_frac + val_frac)))

        train_idx.append(idx_t[:cut_train])
        val_idx.append(idx_t[cut_train:cut_val])
        test_idx.append(idx_t[cut_val:])

    train_idx = np.concatenate(train_idx).astype(np.int64)
    val_idx   = np.concatenate(val_idx).astype(np.int64)
    test_idx  = np.concatenate(test_idx).astype(np.int64)

    X_train, p_train = features[train_idx], prices[train_idx]
    X_val,   p_val   = features[val_idx],   prices[val_idx]
    X_test,  p_test  = features[test_idx],  prices[test_idx]

    seg_train_len = int(np.floor(rows_per_ticker * train_frac))
    seg_val_len   = int(np.floor(rows_per_ticker * val_frac))
    seg_test_len  = rows_per_ticker - seg_train_len - seg_val_len
    n_segs = len(tickers)

    return (
        X_train, p_train,
        X_val, p_val,
        X_test, p_test,
        seg_train_len, seg_val_len, seg_test_len,
        n_segs
    )

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_yaml_to_ns(path: str) -> SimpleNamespace:
    # Minimal YAML loader that returns a dot-accessible namespace.
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    def rec(x):
        if isinstance(x, dict):
            return SimpleNamespace(**{k: rec(v) for k, v in x.items()})
        if isinstance(x, list):
            return [rec(v) for v in x]
        return x

    return rec(data)

def now_run_id(prefix: str = "pipeline") -> str:
    return f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

def check_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)