from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def compute_q_gap(agent, states: np.ndarray, buy_action: int = 1, hold_action: int = 0, max_points: int = 2000) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    n = min(len(states), int(max_points))
    gaps = np.empty(n, dtype=np.float32)
    for i in range(n):
        q = agent.q_values(states[i])
        gaps[i] = float(q[int(buy_action)] - q[int(hold_action)])
    return gaps


def plot_q_gap(gaps: np.ndarray, out_dir: str, tag: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    line_path = os.path.join(out_dir, f"q_gap_{tag}.png")
    hist_path = os.path.join(out_dir, f"q_gap_{tag}_hist.png")

    plt.figure()
    plt.plot(gaps)
    plt.axhline(0.0, linestyle="--")
    plt.title("Q gap: Q(BUY) - Q(HOLD)")
    plt.savefig(line_path)
    plt.close()

    plt.figure()
    plt.hist(gaps, bins=50)
    plt.title("Q gap distribution")
    plt.savefig(hist_path)
    plt.close()

    return {"line": line_path, "hist": hist_path}
