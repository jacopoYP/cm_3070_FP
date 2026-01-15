from __future__ import annotations
import numpy as np


def softmax_confidence(q: np.ndarray, action: int, temp: float) -> float:
    q = np.asarray(q, dtype=np.float32)
    t = max(1e-6, float(temp))
    z = (q - np.max(q)) / t
    exp = np.exp(z)
    probs = exp / (np.sum(exp) + 1e-12)
    return float(probs[int(action)])


def margin_sigmoid_confidence(q: np.ndarray, action: int, temp: float) -> float:
    q = np.asarray(q, dtype=np.float32)
    a = int(action)
    q_best = float(q[a])
    q_alt = float(np.max(np.delete(q, a)))
    margin = q_best - q_alt
    t = max(1e-6, float(temp))
    return float(1.0 / (1.0 + np.exp(-(margin / t))))
