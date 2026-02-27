from __future__ import annotations
import numpy as np

from dataclasses import dataclass
# from typing import Dict, Optional, Tuple
from typing import Dict, Optional

from core.math_utils import safe_divide
from core.helper import net_return

HOLD = 0
SELL = 1


@dataclass
class SellEpisode:
    entry_idx: int
    entry_price: float
    t: int
    bars_held: int


class SellEnv:
    """
    Sell environment:
      - Each episode starts at a BUY entry index (already in position).
      - Agent decides HOLD(0) or SELL(1).
      - Reward is 0 while holding.
      - Terminal reward is net return from entry -> exit minus transaction cost ONCE at exit

    Forced exit at:
      - horizon end (entry_idx + sell_horizon),
      - segment end,
      - end of data.
    """

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        entry_indices: np.ndarray,
        transaction_cost: float = 0.001,
        sell_horizon: int = 20,
        min_hold_bars: int = 10,
        segment_len: Optional[int] = None,
        include_pos_features: bool = True,
        seed: int = 42,
    ):
        self.X = np.asarray(features, dtype=np.float32)
        self.p = np.asarray(prices, dtype=np.float32)
        self.entries = np.asarray(entry_indices, dtype=np.int32)

        self.tc = float(transaction_cost)
        self.horizon = int(sell_horizon)
        self.min_hold = int(min_hold_bars)

        self.segment_len = int(segment_len) if segment_len is not None else None
        self.include_pos = bool(include_pos_features)

        # Safety checks
        self.n = len(self.p)
        if self.X.shape[0] != self.n:
            raise ValueError("features and prices must have same length")
        if len(self.entries) == 0:
            raise ValueError("entry_indices is empty")
        if self.horizon < 0:
            raise ValueError("sell_horizon must be >= 0")

        self.feat_dim = int(self.X.shape[1])
        self.pos_dim = 3 if self.include_pos else 0
        self.state_dim = self.feat_dim + self.pos_dim

        self.rng = np.random.default_rng(int(seed))
        self.ep: Optional[SellEpisode] = None

    # ---------------------------------------------------------------------
    # Segment helpers
    # ---------------------------------------------------------------------

    def _segment_end(self, t: int) -> int:
        if self.segment_len is None:
            return self.n - 1
        seg = int(t // self.segment_len)
        end = (seg + 1) * self.segment_len - 1
        return int(min(end, self.n - 1))

    def _last_allowed(self, entry_idx: int) -> int:
        # Last bar (inclusive) where the position may still be open / exit may occur.

        seg_end = self._segment_end(entry_idx)
        return int(min(entry_idx + self.horizon, seg_end, self.n - 1))

    # ---------------------------------------------------------------------
    # Env API
    # ---------------------------------------------------------------------

    def reset(self, entry_idx: Optional[int] = None) -> np.ndarray:
        if entry_idx is None:
            entry_idx = int(self.rng.choice(self.entries))

        entry_idx = int(entry_idx)
        if entry_idx < 0 or entry_idx >= self.n:
            raise ValueError("entry_idx out of range")

        self.ep = SellEpisode(
            entry_idx=entry_idx,
            entry_price=float(self.p[entry_idx]),
            t=entry_idx,
            bars_held=0,
        )
        return self._get_state()

    def step(self, action: int):
        if self.ep is None:
            raise RuntimeError("Call reset() before step().")

        action = int(action)
        t = int(self.ep.t)
        entry_idx = int(self.ep.entry_idx)
        last_allowed = int(self._last_allowed(entry_idx))
        bars_held = int(self.ep.bars_held)

        can_sell = bars_held >= self.min_hold

        # Info dictionary 
        info: Dict = {
            "t": t,
            "entry_idx": entry_idx,
            "bars_held": bars_held,
            "last_allowed": last_allowed,
        }

        # Baseline: what happens if we simply hold until the horizon/segment end
        baseline = float(self._net_tm_at(last_allowed))
        info["baseline_net_tm"] = baseline

        # Forced exit at last_allowed: by definition delta = 0
        if t >= last_allowed:
            rets = self._returns_at(t)
            info.update({"forced_exit": True, "reason": "limit", "exit_idx": int(t), **rets})
            info["delta_vs_hold"] = 0.0
            return self._terminal_state(), 0.0, True, info

        # Voluntary SELL: reward is improvement vs holding to last_allowed
        if action == SELL and can_sell:
            net_now = float(self._net_tm_at(t))
            delta = net_now - baseline
            rets = self._returns_at(t)
            info.update({"forced_exit": False, "reason": "sell", "exit_idx": int(t), **rets})
            info["delta_vs_hold"] = float(delta)
            return self._terminal_state(), float(delta), True, info

        # HOLD (or SELL not allowed yet): reward 0, advance time
        self.ep.t = t + 1
        self.ep.bars_held += 1
        return self._get_state(), 0.0, False, info

    # ---------------------------------------------------------------------
    # Utils methods
    # ---------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """
        State at current time t (in position).
        If include_pos_features, it appends:
          - unrealized_return
          - time_frac in [0,1]
          - remaining_frac in [0,1]
        """
        assert self.ep is not None
        t = int(self.ep.t)
        entry_idx = int(self.ep.entry_idx)
        entry_price = float(self.ep.entry_price)

        base = self.X[t].astype(np.float32, copy=False)

        if not self.include_pos:
            return base

        last_allowed = self._last_allowed(entry_idx)
        price_now = float(self.p[t])

        # unreal = (price_now - entry_price) / (entry_price + 1e-12)
        unreal = safe_divide(price_now - entry_price, entry_price)

        eff_h = max(1, int(last_allowed - entry_idx))
        time_frac = float(min(1.0, self.ep.bars_held / eff_h))
        remaining = float(max(0, last_allowed - t))
        remaining_frac = float(min(1.0, remaining / eff_h))

        extra = np.array([unreal, time_frac, remaining_frac], dtype=np.float32)
        return np.concatenate([base, extra], axis=0)

    def _terminal_state(self) -> np.ndarray:
        return np.zeros((self.state_dim,), dtype=np.float32)
    
    
    def _returns_at(self, exit_idx: int) -> Dict[str, float]:
        assert self.ep is not None
        tc = float(self.tc)
        entry_price = float(self.ep.entry_price)
        exit_price = float(self.p[exit_idx])

        # gross = (exit_price - entry_price) / (entry_price + 1e-12)
        gross = safe_divide(exit_price - entry_price, entry_price)

        # Gross minus 'exit' cost only
        net_env = gross - tc

        # What TradeManager reports: round-trip exact net
        net_tm = ((1.0 - tc) * (1.0 - tc) * (1.0 + gross)) - 1.0

        return {
            "exit_price": exit_price,
            "gross_return": float(gross),
            "net_return_env": float(net_env),
            "net_return_tm": float(net_tm),
        }
    
    # def _net_tm_at(self, exit_idx: int) -> float:
    #     """TradeManager-consistent round-trip net return at exit_idx (entry+exit costs)."""
    #     assert self.ep is not None
    #     tc = float(self.tc)
    #     entry_price = float(self.ep.entry_price)
    #     exit_price = float(self.p[exit_idx])
    #     # gross = (exit_price - entry_price) / (entry_price + 1e-12)
    #     gross = safe_divide(exit_price - entry_price, entry_price)

    #     return ((1.0 - tc) * (1.0 - tc) * (1.0 + gross)) - 1.0

    def _net_tm_at(self, exit_idx: int) -> float:
        assert self.ep is not None
        return net_return(
            exit_price=float(self.p[exit_idx]),
            entry_price=float(self.ep.entry_price),
            tc=float(self.tc),
        )
