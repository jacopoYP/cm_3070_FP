from __future__ import annotations
import numpy as np


# def softmax_confidence(q: np.ndarray, action: int, temp: float) -> float:
#     q = np.asarray(q, dtype=np.float32)
#     t = max(1e-6, float(temp))
#     z = (q - np.max(q)) / t
#     exp = np.exp(z)
#     probs = exp / (np.sum(exp) + 1e-12)
#     return float(probs[int(action)])


# def margin_sigmoid_confidence(q: np.ndarray, action: int, temp: float) -> float:
#     q = np.asarray(q, dtype=np.float32)
#     a = int(action)
#     q_best = float(q[a])
#     q_alt = float(np.max(np.delete(q, a)))
#     margin = q_best - q_alt
#     t = max(1e-6, float(temp))
#     return float(1.0 / (1.0 + np.exp(-(margin / t))))

import numpy as np
from core.math_utils import MIN_TEMP

def softmax_confidence(q: np.ndarray, action: int, temp: float) -> float:
    q = np.asarray(q, dtype=np.float64)
    a = int(action)

    t = float(np.clip(temp, MIN_TEMP, np.inf))
    z = (q - np.max(q)) / t
    exp = np.exp(z)

    s = exp.sum()
    # Degenerate fallback (rare): return uniform
    if s == 0.0 or not np.isfinite(s):
        return float(1.0 / q.size)

    probs = exp / s
    return float(probs[a])


def margin_sigmoid_confidence(q: np.ndarray, action: int, temp: float) -> float:
    q = np.asarray(q, dtype=np.float64)
    a = int(action)

    q_best = float(q[a])
    q_alt = float(np.max(np.delete(q, a))) if q.size > 1 else q_best
    margin = q_best - q_alt

    t = float(np.clip(temp, MIN_TEMP, np.inf))
    x = margin / t

    # avoid overflow in exp for extreme values
    x = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x)))

