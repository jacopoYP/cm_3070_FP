from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.types import RewardConfig, TradeManagerConfig

from core.helper import check_sentiment

@dataclass
class BuyEnvState:
    t: int = 0
    in_pos: bool = False
    entry_price: float = 0.0
    entry_t: int = 0
    cooldown: int = 0
    done: bool = False


class BuyEnv:
    """Buy-only environment.

    Actions:
        0 = HOLD
        1 = BUY

    Position closes via:
        - time stop (sell_horizon)
        - end of data
    """

    HOLD = 0
    BUY = 1

    def __init__(self, features: np.ndarray, prices: np.ndarray, reward: RewardConfig, trade: TradeManagerConfig):
        if len(features) != len(prices):
            raise ValueError("features and prices must have same length")
        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.reward_cfg = reward
        self.trade_cfg = trade
        self.state_dim = int(features.shape[1])
        self.reset()

    # ---------------------------------------------------------------------
    # Env API
    # ---------------------------------------------------------------------
    
    def reset(self) -> np.ndarray:
        self.s = BuyEnvState()
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.s.done:
            raise RuntimeError("step() called after done=True")

        info: Dict[str, Any] = {}
        r = 0.0

        price = float(self.prices[self.s.t])

        # Cooldown overrides action
        if self.s.cooldown > 0:
            action = self.HOLD
            self.s.cooldown -= 1

        # Siscourage HOLD forever while flat
        if action == self.HOLD and not self.s.in_pos:
            r -= float(self.reward_cfg.flat_hold_penalty)

        # Entry
        if action == self.BUY and not self.s.in_pos:
            if action == self.BUY and not self.s.in_pos:
                if not check_sentiment(self.features[self.s.t], self.trade_cfg):
                    action = self.HOLD
                    info.update({"buy_blocked": "sentiment"})
                else:
                    self.s.in_pos = True
                    self.s.entry_price = price
                    self.s.entry_t = self.s.t
                    r += float(self.reward_cfg.entry_bonus)
                    info.update({"opened": True, "entry_t": self.s.entry_t, "entry_price": self.s.entry_price})

        # In-position shaping
        if action == self.HOLD and self.s.in_pos:
            r -= float(self.reward_cfg.in_pos_hold_penalty)

        # Time stop
        if self.s.in_pos:
            hold_bars = int(self.s.t - self.s.entry_t)
            if hold_bars >= int(self.trade_cfg.sell_horizon):
                r_close, tinfo = self._close(price, forced=True, reason="time")
                r += r_close
                info.update(tinfo)

        # Advance time
        self.s.t += 1
        if self.s.t >= len(self.prices) - 1:
            self.s.done = True
            if self.s.in_pos:
                last_price = float(self.prices[self.s.t])
                r_close, tinfo = self._close(last_price, forced=True, reason="eod")
                r += r_close
                info.update(tinfo)

        return self._obs(), float(r), bool(self.s.done), info

    # ---------------------------------------------------------------------
    # Utils methods
    # ---------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        t = min(self.s.t, len(self.features) - 1)
        return self.features[t]

    def _close(self, exit_price: float, forced: bool, reason: str):
        gross = (exit_price - float(self.s.entry_price)) / float(self.s.entry_price)
        
        # Considering transaction cost
        net = gross - 2.0 * float(self.reward_cfg.transaction_cost)

        # reward : return - dd penalty - vol penalty
        rew = net
        rew -= float(self.reward_cfg.lambda_dd) * max(0.0, -net)
        rew -= float(self.reward_cfg.lambda_vol) * (net ** 2)

        entry_t = int(self.s.entry_t)
        hold_bars = int(self.s.t - entry_t)

        self.s.in_pos = False
        self.s.entry_price = 0.0
        self.s.entry_t = 0
        self.s.cooldown = int(self.trade_cfg.cooldown_steps)

        return float(rew), {
            "closed": True,
            "reason": reason,
            "forced_exit": bool(forced),
            "entry_t": entry_t,
            "exit_t": int(self.s.t),
            "hold_bars": hold_bars,
            "gross_return": float(gross),
            "net_return": float(net),
        }
