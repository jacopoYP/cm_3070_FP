from typing import Any, Dict, List
import numpy as np

def summarize_trades(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trades:
        return {
            "n_trades": 0,
            "avg_net_return": 0.0,
            "win_rate": 0.0,
            "min_net": 0.0,
            "median_net": 0.0,
            "max_net": 0.0,
        }

    net = np.array([t.get("net_return", 0.0) for t in trades], dtype=float)
    return {
        "n_trades": int(len(trades)),
        "avg_net_return": float(np.mean(net)),
        "win_rate": float(np.mean(net > 0)),
        "min_net": float(np.min(net)),
        "median_net": float(np.median(net)),
        "max_net": float(np.max(net)),
    }