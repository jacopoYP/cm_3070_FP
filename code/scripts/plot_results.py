#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import yaml

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# -----------------------------
# IO helpers
# -----------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -----------------------------
# Trade parsing + derived series
# -----------------------------

def trades_to_arrays(trades: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Trades are emitted by TradeManager as dicts like:
      {
        "entry_idx": int,
        "exit_idx": int,
        "net_return": float,
        "gross_return": float,
        "hold_bars": int,
        "forced_exit": bool,
        "meta": {"reason": "...", "buy_conf": ... , ...}
      }
    """
    if not trades:
        return {
            "entry_idx": np.array([], dtype=int),
            "exit_idx": np.array([], dtype=int),
            "net_return": np.array([], dtype=float),
            "gross_return": np.array([], dtype=float),
            "hold_bars": np.array([], dtype=int),
            "forced_exit": np.array([], dtype=bool),
            "reason": np.array([], dtype=str),
        }

    entry_idx = np.array([int(t.get("entry_idx", -1)) for t in trades], dtype=int)
    exit_idx = np.array([int(t.get("exit_idx", -1)) for t in trades], dtype=int)
    net_return = np.array([float(t.get("net_return", 0.0)) for t in trades], dtype=float)
    gross_return = np.array([float(t.get("gross_return", 0.0)) for t in trades], dtype=float)
    hold_bars = np.array([int(t.get("hold_bars", 0)) for t in trades], dtype=int)
    forced_exit = np.array([bool(t.get("forced_exit", False)) for t in trades], dtype=bool)

    reasons: List[str] = []
    for t in trades:
        meta = t.get("meta", {}) or {}
        reasons.append(str(meta.get("reason", "unknown")))
    reason = np.array(reasons, dtype=str)

    return {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "net_return": net_return,
        "gross_return": gross_return,
        "hold_bars": hold_bars,
        "forced_exit": forced_exit,
        "reason": reason,
    }

def equity_from_trade_net_returns(trades: List[Dict[str, Any]], start_equity: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds a simple trade-by-trade equity curve using each trade's net_return:
      equity_{k+1} = equity_k * (1 + net_return_k)

    Note: This is a trade-level curve (not bar-by-bar). It matches your JSON logs easily and
    is perfect for report plots.
    """
    if not trades:
        return np.array([0], dtype=int), np.array([start_equity], dtype=float)

    # sort by entry_idx for consistency
    trades_sorted = sorted(trades, key=lambda x: int(x.get("entry_idx", 0)))
    net = np.array([float(t.get("net_return", 0.0)) for t in trades_sorted], dtype=float)
    x = np.arange(len(net) + 1, dtype=int)

    eq = np.empty(len(net) + 1, dtype=float)
    eq[0] = float(start_equity)
    for i in range(len(net)):
        eq[i + 1] = eq[i] * (1.0 + net[i])

    return x, eq

def reason_counts(reason: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in reason.tolist():
        out[r] = out.get(r, 0) + 1
    return out


# -----------------------------
# Feature/sentiment helpers
# -----------------------------

@dataclass
class SentimentColumns:
    score_idx: int
    mass_idx: int

def infer_sentiment_columns(n_features: int) -> SentimentColumns:
    """
    Your pipeline appends sentiment_score and sentiment_mass as the last 2 features (most common).
    If you ever change that, just modify here or pass explicit indices via args later.
    """
    if n_features < 2:
        raise ValueError("Need at least 2 features to infer sentiment columns.")
    return SentimentColumns(score_idx=n_features - 2, mass_idx=n_features - 1)

def slice_first_segment(x: np.ndarray, seg_len: int) -> np.ndarray:
    return x[: min(seg_len, len(x))]


# -----------------------------
# GA log helpers
# -----------------------------

def ga_best_mean_by_generation(ga_rows: List[Dict[str, Any]], pop_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    We assume the GA evaluator logs one JSON line per evaluated individual.
    If gens=G and pop=P, the log is typically ~G*P lines (maybe + elites, depending on implementation).
    We keep it simple: chunk sequentially in groups of pop_size.

    Each row should contain "fitness" OR some "test_final_equity" field.
    """
    if not ga_rows or pop_size <= 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    # fitness key fallback
    def get_fit(r: Dict[str, Any]) -> float:
        for k in ["fitness", "test_fitness", "test_final_equity", "final_equity"]:
            if k in r:
                try:
                    return float(r[k])
                except Exception:
                    pass
        # if nested
        t = r.get("test", {}) or {}
        if "final_equity" in t:
            return float(t["final_equity"])
        return float("nan")

    fits = np.array([get_fit(r) for r in ga_rows], dtype=float)
    fits = fits[np.isfinite(fits)]
    if len(fits) == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    n_chunks = int(np.ceil(len(fits) / pop_size))
    best = []
    mean = []
    gen = []
    for g in range(n_chunks):
        chunk = fits[g * pop_size : (g + 1) * pop_size]
        if len(chunk) == 0:
            continue
        gen.append(g)
        best.append(float(np.max(chunk)))
        mean.append(float(np.mean(chunk)))

    return np.array(gen, dtype=int), np.array(best, dtype=float), np.array(mean, dtype=float)


# -----------------------------
# Plotters
# -----------------------------

def plot_equity_curves(out_dir: str,
                       train_buy_only: Optional[List[Dict[str, Any]]],
                       train_with_sell: Optional[List[Dict[str, Any]]],
                       test_buy_only: Optional[List[Dict[str, Any]]],
                       test_with_sell: Optional[List[Dict[str, Any]]]) -> None:
    plt.figure()
    if train_buy_only is not None:
        x, eq = equity_from_trade_net_returns(train_buy_only)
        plt.plot(x, eq, label="TRAIN buy_only")
    if train_with_sell is not None:
        x, eq = equity_from_trade_net_returns(train_with_sell)
        plt.plot(x, eq, label="TRAIN with_sell")
    if test_buy_only is not None:
        x, eq = equity_from_trade_net_returns(test_buy_only)
        plt.plot(x, eq, label="TEST buy_only")
    if test_with_sell is not None:
        x, eq = equity_from_trade_net_returns(test_with_sell)
        plt.plot(x, eq, label="TEST with_sell")

    plt.title("Trade-level equity curves (from net_return)")
    plt.xlabel("Trade count")
    plt.ylabel("Equity (starting at 1.0)")
    plt.legend()
    save_fig(os.path.join(out_dir, "equity_curves.png"))

def plot_return_histograms(out_dir: str,
                           buy_only: Optional[List[Dict[str, Any]]],
                           with_sell: Optional[List[Dict[str, Any]]],
                           title: str,
                           fname: str) -> None:
    plt.figure()
    if buy_only is not None and len(buy_only) > 0:
        a = trades_to_arrays(buy_only)["net_return"]
        plt.hist(a, bins=30, alpha=0.6, label="buy_only")
    if with_sell is not None and len(with_sell) > 0:
        b = trades_to_arrays(with_sell)["net_return"]
        plt.hist(b, bins=30, alpha=0.6, label="with_sell")

    plt.title(title)
    plt.xlabel("Trade net_return")
    plt.ylabel("Count")
    plt.legend()
    save_fig(os.path.join(out_dir, fname))

def plot_exit_reasons(out_dir: str,
                      trades: Optional[List[Dict[str, Any]]],
                      title: str,
                      fname: str) -> None:
    if not trades:
        return
    arr = trades_to_arrays(trades)
    counts = reason_counts(arr["reason"])
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Exit reason")
    plt.ylabel("Count")
    save_fig(os.path.join(out_dir, fname))

def plot_sentiment_and_price(out_dir: str,
                             features: np.ndarray,
                             prices: np.ndarray,
                             seg_len: int) -> None:
    # first segment only (keeps plots interpretable)
    X0 = slice_first_segment(features, seg_len)
    p0 = slice_first_segment(prices, seg_len)

    cols = infer_sentiment_columns(X0.shape[1])
    sent = X0[:, cols.score_idx]
    mass = X0[:, cols.mass_idx]
    t = np.arange(len(p0))

    # price
    plt.figure()
    plt.plot(t, p0)
    plt.title("Price (first segment)")
    plt.xlabel("t (bars)")
    plt.ylabel("Price")
    save_fig(os.path.join(out_dir, "price_first_segment.png"))

    # sentiment score
    plt.figure()
    plt.plot(t, sent)
    plt.title("Sentiment score (first segment)")
    plt.xlabel("t (bars)")
    plt.ylabel("Sentiment score")
    save_fig(os.path.join(out_dir, "sentiment_score_first_segment.png"))

    # sentiment mass
    plt.figure()
    plt.plot(t, mass)
    plt.title("Sentiment mass (first segment)")
    plt.xlabel("t (bars)")
    plt.ylabel("Sentiment mass")
    save_fig(os.path.join(out_dir, "sentiment_mass_first_segment.png"))

def plot_ga_progress(out_dir: str,
                     ga_log_path: str,
                     meta_path: Optional[str] = None,
                     pop_override: Optional[int] = None) -> None:
    if not os.path.exists(ga_log_path):
        return

    ga_rows = load_jsonl(ga_log_path)

    pop_size = pop_override
    if pop_size is None and meta_path and os.path.exists(meta_path):
        meta = load_json(meta_path)
        pop_size = int(meta.get("pop", meta.get("population", 0)) or 0)
    if not pop_size:
        # fallback: assume 16 if unknown
        pop_size = 16

    gen, best, mean = ga_best_mean_by_generation(ga_rows, pop_size=pop_size)
    if len(gen) == 0:
        return

    plt.figure()
    plt.plot(gen, best, label="best fitness")
    plt.plot(gen, mean, label="mean fitness")
    plt.title("GA fitness progression (chunked by population size)")
    plt.xlabel("Generation (approx)")
    plt.ylabel("Fitness (e.g., test final_equity)")
    plt.legend()
    save_fig(os.path.join(out_dir, "ga_fitness_progress.png"))

def plot_buyhold_benchmark(out_dir: str,
                           prices: np.ndarray,
                           tickers: List[str],
                           symbols: List[str],
                           rows_per_ticker: int,
                           train_len: int,
                           val_len: int,
                           test_len: int,
                           title: str = "Buy & Hold benchmark (normalized equity)") -> None:
    """
    Plots buy&hold equity curves (normalized to 1.0 at test start) for selected symbols
    using the segment layout: [ticker0 rows][ticker1 rows]...[tickerN rows].

    prices: flat array (N,), N = len(tickers) * rows_per_ticker
    """
    if prices.ndim != 1:
        raise ValueError(f"Expected prices as 1D array, got shape={prices.shape}")

    needed = len(tickers) * rows_per_ticker
    if len(prices) < needed:
        raise ValueError(f"prices too short: len(prices)={len(prices)} < {needed} (=tickers*rows_per_ticker)")

    start_off = train_len + val_len
    if start_off + test_len > rows_per_ticker:
        raise ValueError("train_len+val_len+test_len exceeds rows_per_ticker")

    plt.figure()
    for sym in symbols:
        if sym not in tickers:
            print(f"[WARN] symbol not in tickers list: {sym}")
            continue

        idx = tickers.index(sym)
        seg_start = idx * rows_per_ticker
        seg_end = seg_start + rows_per_ticker
        seg = prices[seg_start:seg_end]

        p = seg[start_off : start_off + test_len].astype(float, copy=False)
        if len(p) < 2:
            print(f"[WARN] not enough prices for {sym} in selected window")
            continue

        # normalized equity curve
        p0 = max(float(p[0]), 1e-12)
        eq = p / p0
        t = np.arange(len(eq), dtype=int)

        plt.plot(t, eq, label=f"{sym} (B&H)")

    plt.title(title)
    plt.xlabel("t (bars) [test window]")
    plt.ylabel("Equity (normalized to 1.0)")
    plt.legend()
    save_fig(os.path.join(out_dir, "buyhold_benchmark.png"))


# -----------------------------
# Text summary for report
# -----------------------------

def write_quick_summary(out_dir: str, summary: Optional[Dict[str, Any]]) -> None:
    if not summary:
        return
    lines: List[str] = []

    def add_block(name: str, block: Dict[str, Any]) -> None:
        lines.append(f"\n{name}")
        for k in ["n_trades", "final_equity", "avg_net_return", "win_rate", "min_net", "median_net", "max_net"]:
            if k in block:
                lines.append(f"  - {k}: {block[k]}")
        if "exit_reasons" in block:
            lines.append(f"  - exit_reasons: {block['exit_reasons']}")
        if "entry_debug" in block:
            ed = block["entry_debug"]
            pick = {kk: ed.get(kk) for kk in ["checked","opened","blocked_sentiment","blocked_conf","blocked_trend","blocked_latest_entry"] if isinstance(ed, dict)}
            lines.append(f"  - entry_debug: {pick}")

    # try to match your run_pipeline summary structure
    train = summary.get("train", {}) if isinstance(summary, dict) else {}
    test = summary.get("test", {}) if isinstance(summary, dict) else {}

    if "tm_buy_only" in train:
        add_block("TRAIN tm_buy_only", train["tm_buy_only"])
    if "tm_with_sell" in train:
        add_block("TRAIN tm_with_sell", train["tm_with_sell"])
    if "tm_buy_only" in test:
        add_block("TEST tm_buy_only", test["tm_buy_only"])
    if "tm_with_sell" in test:
        add_block("TEST tm_with_sell", test["tm_with_sell"])

    path = os.path.join(out_dir, "plots_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


# -----------------------------
# main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=None,
                    help="Pipeline run dir (contains summary.json and trades_*.json).")
    ap.add_argument("--summary", type=str, default=None, help="Path to summary.json (optional).")

    ap.add_argument("--train_buy_only", type=str, default=None)
    ap.add_argument("--test_buy_only", type=str, default=None)
    ap.add_argument("--train_with_sell", type=str, default=None)
    ap.add_argument("--test_with_sell", type=str, default=None)

    ap.add_argument("--features", type=str, default=None, help="features.npy (for sentiment plots).")
    ap.add_argument("--prices", type=str, default=None, help="prices.npy (for sentiment/price plots).")
    ap.add_argument("--seg_len", type=int, default=1239, help="Segment length (for first-segment plots).")

    ap.add_argument("--ga_dir", type=str, default=None, help="GA run dir containing ga_log.jsonl and meta.json.")
    ap.add_argument("--ga_log", type=str, default=None, help="Explicit ga_log.jsonl path (optional).")
    ap.add_argument("--ga_meta", type=str, default=None, help="Explicit meta.json path (optional).")
    ap.add_argument("--ga_pop", type=int, default=None, help="Override population size for chunking.")

    # SPY Comparison
    ap.add_argument("--config", type=str, default=None, help="config.yaml (to read tickers list for benchmark plot)")
    ap.add_argument("--bench_symbols", type=str, default="AAPL,SPY", help="Comma-separated symbols to compare (e.g. AAPL,SPY)")
    ap.add_argument("--rows_per_ticker", type=int, default=1238, help="Rows per ticker segment (must match your dataset layout)")
    ap.add_argument("--train_len", type=int, default=866)
    ap.add_argument("--val_len", type=int, default=185)
    ap.add_argument("--test_len", type=int, default=187)


    ap.add_argument("--out_dir", type=str, default=None, help="Where to write plots (default: <run_dir>/plots).")
    args = ap.parse_args()

    # Infer defaults from run_dir
    run_dir = args.run_dir
    if run_dir:
        if args.summary is None:
            cand = os.path.join(run_dir, "summary.json")
            if os.path.exists(cand):
                args.summary = cand

        for name in ["train_buy_only", "test_buy_only", "train_with_sell", "test_with_sell"]:
            if getattr(args, name) is None:
                # common filenames you used
                mapping = {
                    "train_buy_only": "trades_train_buy_only.json",
                    "test_buy_only": "trades_test_buy_only.json",
                    "train_with_sell": "trades_train_with_sell.json",
                    "test_with_sell": "trades_test_with_sell.json",
                }
                cand = os.path.join(run_dir, mapping[name])
                if os.path.exists(cand):
                    setattr(args, name, cand)

        if args.out_dir is None:
            args.out_dir = os.path.join(run_dir, "plots")

    if args.out_dir is None:
        args.out_dir = "plots"

    ensure_dir(args.out_dir)

    summary = load_json(args.summary) if args.summary and os.path.exists(args.summary) else None

    # Load trades
    def load_trades_maybe(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        if not path:
            return None
        if not os.path.exists(path):
            return None
        return load_json(path)

    train_buy_only = load_trades_maybe(args.train_buy_only)
    test_buy_only  = load_trades_maybe(args.test_buy_only)
    train_with_sell = load_trades_maybe(args.train_with_sell)
    test_with_sell  = load_trades_maybe(args.test_with_sell)

    # Plots: equity + hist + exit reasons
    plot_equity_curves(args.out_dir, train_buy_only, train_with_sell, test_buy_only, test_with_sell)
    plot_return_histograms(args.out_dir, train_buy_only, train_with_sell,
                           title="TRAIN net_return distribution (buy_only vs with_sell)",
                           fname="hist_train_net_returns.png")
    plot_return_histograms(args.out_dir, test_buy_only, test_with_sell,
                           title="TEST net_return distribution (buy_only vs with_sell)",
                           fname="hist_test_net_returns.png")

    plot_exit_reasons(args.out_dir, train_with_sell, "TRAIN exit reasons (with_sell)", "exit_reasons_train_with_sell.png")
    plot_exit_reasons(args.out_dir, test_with_sell,  "TEST exit reasons (with_sell)",  "exit_reasons_test_with_sell.png")

    # Sentiment + price (optional)
    if args.features and args.prices and os.path.exists(args.features) and os.path.exists(args.prices):
        X = np.load(args.features).astype(np.float32, copy=False)
        p = np.load(args.prices).astype(np.float32, copy=False)
        if len(X) == len(p):
            plot_sentiment_and_price(args.out_dir, X, p, seg_len=int(args.seg_len))
        else:
            print(f"[WARN] features/prices length mismatch: {X.shape} vs {p.shape}")

        # Benchmark plot: AAPL vs SPY (buy&hold), using test window
    if args.config and args.prices and os.path.exists(args.config) and os.path.exists(args.prices):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # adjust this path if your config stores tickers elsewhere
        tickers = cfg["data"]["tickers"]

        p_all = np.load(args.prices).astype(np.float32, copy=False)
        symbols = [s.strip() for s in args.bench_symbols.split(",") if s.strip()]

        plot_buyhold_benchmark(
            out_dir=args.out_dir,
            prices=p_all,
            tickers=tickers,
            symbols=symbols,
            rows_per_ticker=int(args.rows_per_ticker),
            train_len=int(args.train_len),
            val_len=int(args.val_len),
            test_len=int(args.test_len),
            title="Buy & Hold benchmark (TEST): "+str(symbols)
        )


    # GA plots (optional)
    ga_log = args.ga_log
    ga_meta = args.ga_meta
    if args.ga_dir:
        if ga_log is None:
            cand = os.path.join(args.ga_dir, "ga_log.jsonl")
            if os.path.exists(cand):
                ga_log = cand
        if ga_meta is None:
            cand = os.path.join(args.ga_dir, "meta.json")
            if os.path.exists(cand):
                ga_meta = cand

    if ga_log and os.path.exists(ga_log):
        plot_ga_progress(args.out_dir, ga_log_path=ga_log, meta_path=ga_meta, pop_override=args.ga_pop)

    # Write a compact text summary (handy to paste into Evaluation)
    write_quick_summary(args.out_dir, summary)

    print("\nSaved plots to:", args.out_dir)
    for fn in sorted(os.listdir(args.out_dir)):
        if fn.lower().endswith(".png") or fn.lower().endswith(".txt"):
            print(" -", fn)


if __name__ == "__main__":
    main()
