import numpy as np

def summarize_backtest(res: dict) -> dict:
    trades = res["trades"]
    eq = np.asarray(res["equity_curve"], dtype=np.float32)

    out = {
        "final_equity": float(res["final_equity"]),
        "n_trades": int(len(trades)),
    }

    if len(trades) == 0:
        out.update({
            "win_rate": 0.0,
            "avg_net_return": 0.0,
            "median_net_return": 0.0,
            "avg_hold_bars": 0.0,
            "forced_time_pct": 0.0,
            "forced_eod_pct": 0.0,
            "max_drawdown": 0.0,
        })
        return out

    net = np.array([t["net_return"] for t in trades], dtype=np.float32)
    hold = np.array([t.get("hold_bars", t["exit_idx"] - t["entry_idx"]) for t in trades], dtype=np.int32)

    win_rate = float((net > 0).mean())
    avg_net = float(net.mean())
    med_net = float(np.median(net))
    avg_hold = float(hold.mean())

    forced_time = sum(1 for t in trades if t.get("forced_exit") and t.get("reason") == "time")
    forced_eod  = sum(1 for t in trades if t.get("forced_exit") and t.get("reason") == "eod")
    forced_any  = sum(1 for t in trades if t.get("forced_exit"))

    # Max drawdown from equity curve
    peak = np.maximum.accumulate(eq)
    dd = (eq / (peak + 1e-12)) - 1.0
    max_dd = float(dd.min())

    out.update({
        "win_rate": win_rate,
        "avg_net_return": avg_net,
        "median_net_return": med_net,
        "avg_hold_bars": avg_hold,
        "forced_any_pct": float(forced_any / len(trades)),
        "forced_time_pct": float(forced_time / len(trades)),
        "forced_eod_pct": float(forced_eod / len(trades)),
        "max_drawdown": max_dd,
    })
    return out

def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Compute maximum drawdown from an equity curve.

    Returns a negative number (e.g. -0.23 = -23%).
    """
    if equity_curve is None:
        return 0.0

    curve = np.asarray(equity_curve, dtype=np.float64)

    if curve.size < 2:
        return 0.0

    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())
