# agents/utils/entry_indices.py
import numpy as np

def generate_buy_entry_indices(
    buy_agent,
    state_df: np.ndarray,         # (T, D) float32
    prices: np.ndarray,           # (T,) float32
    *,
    buy_min_confidence: float | None = None,
    confidence_temp: float = 0.01,
    use_trend_filter: bool = False,
    ma_short: int = 10,
    ma_long: int = 30,
    cooldown_steps: int = 0,
) -> np.ndarray:
    assert state_df.ndim == 2
    assert prices.ndim == 1 and len(prices) == len(state_df)

    T = len(prices)

    # Trend mask
    trend_ok = np.ones(T, dtype=bool)
    if use_trend_filter:
        trend_ok[:] = False
        def sma(x, w):
            if w <= 1:
                return x.astype(np.float32)
            c = np.cumsum(np.insert(x.astype(np.float32), 0, 0.0))
            out = (c[w:] - c[:-w]) / float(w)
            return np.concatenate([np.full(w - 1, np.nan, dtype=np.float32), out.astype(np.float32)])

        ms = sma(prices, ma_short)
        ml = sma(prices, ma_long)
        valid = ~np.isnan(ms) & ~np.isnan(ml)
        trend_ok[valid] = ms[valid] > ml[valid]

    def conf_from_q(q: np.ndarray) -> tuple[int, float, float]:
        action = int(np.argmax(q))
        buy_idx = 1
        q_buy = float(q[buy_idx])
        q_other = float(np.max(np.delete(q, buy_idx))) if q.shape[0] > 1 else q_buy
        raw_margin = q_buy - q_other
        scaled = raw_margin / max(1e-8, confidence_temp)
        conf = 1.0 / (1.0 + np.exp(-scaled))
        return action, float(conf), float(raw_margin)

    entries = []
    cooldown = 0

    for t in range(T):
        if cooldown > 0:
            cooldown -= 1
            continue

        s = state_df[t]

        if hasattr(buy_agent, "q_values"):
            q = np.asarray(buy_agent.q_values(s), dtype=np.float32)
            action, conf, _ = conf_from_q(q)
        else:
            action = int(buy_agent.select_action(s, greedy=True))
            conf = None

        if action != 1:
            continue
        if use_trend_filter and not trend_ok[t]:
            continue
        if buy_min_confidence is not None and conf is not None and conf < buy_min_confidence:
            continue

        entries.append(t)
        cooldown = int(cooldown_steps)

    return np.asarray(entries, dtype=np.int64)

def collect_buy_signal_indices(
    buy_agent,
    state_df: np.ndarray,
    prices: np.ndarray,
    buy_min_confidence: float | None = None,
    use_trend_filter: bool = False,
    ma_short: int = 10,
    ma_long: int = 30,
    confidence_temp: float = 0.01,
    min_gap: int = 5,
) -> np.ndarray:
    n = len(prices)

    if use_trend_filter:
        ma_s = np.convolve(prices, np.ones(ma_short) / ma_short, mode="same")
        ma_l = np.convolve(prices, np.ones(ma_long) / ma_long, mode="same")
        trend_mask = ma_s > ma_l
    else:
        trend_mask = np.ones(n, dtype=bool)

    out = []
    last = -10**9

    for t in range(n):
        if (t - last) < min_gap:
            continue
        if not trend_mask[t]:
            continue

        s = state_df[t].astype(np.float32, copy=False)
        q = np.asarray(buy_agent.q_values(s), dtype=np.float32)
        a = int(np.argmax(q))
        if a != 1:
            continue

        if buy_min_confidence is not None:
            q_buy = float(q[1])
            q_hold = float(q[0])
            margin = q_buy - q_hold
            conf = 1.0 / (1.0 + np.exp(-(margin / max(1e-8, confidence_temp))))
            if conf < buy_min_confidence:
                continue

        out.append(t)
        last = t

    return np.asarray(out, dtype=int)
