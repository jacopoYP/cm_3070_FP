import numpy as np

EPS = np.finfo(np.float32).eps
MIN_TEMP = 1e-6
MIN_STD = 1e-8

def safe_divide(numerator, denominator, fill_value=0.0):
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, fill_value, dtype=float),
        where=denominator != 0,
    )

def clamp_prob(p):
    """
    Clamp probability into (EPS, 1].
    """
    return float(np.clip(p, EPS, 1.0))
