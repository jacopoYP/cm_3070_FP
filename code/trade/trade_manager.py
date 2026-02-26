from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.types import RewardConfig, TradeManagerConfig
from trade.confidence import softmax_confidence, margin_sigmoid_confidence

from core.helper import check_sentiment

@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    hold_bars: int
    forced_exit: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    entry_idx: int
    entry_price: float
    meta: Dict[str, Any] = field(default_factory=dict)


class TradeManager:
    """
    Trade orchestrator.

    - Buy agent opens trades
    - Exit uses forced last_allowed (min(entry+horizon, seg_end(entry), eod))
    - Produces entry_indices for SellAgent training

    - If sell_agent is provided, it can close positions early (aligned with SellEnv)
    """

    def __init__(
        self,
        buy_agent,
        state: np.ndarray,
        prices: np.ndarray,
        reward: RewardConfig,
        trade: TradeManagerConfig,
        sell_agent=None,  # optional
        segment_len: Optional[int] = None,
    ):
        if len(state) != len(prices):
            raise ValueError("state and prices must have the same length")

        self.buy_agent = buy_agent
        self.sell_agent = sell_agent

        self.segment_len = int(segment_len) if segment_len is not None else None

        self.state = np.asarray(state, dtype=np.float32)
        self.prices = np.asarray(prices, dtype=np.float32)

        self.reward_cfg = reward
        self.trade_cfg = trade

        # Precompute MAs
        self._ma_short = None
        self._ma_long = None
        if self.trade_cfg.use_trend_filter:
            self._ma_short = self._sma(self.prices, int(self.trade_cfg.ma_short))
            self._ma_long = self._sma(self.prices, int(self.trade_cfg.ma_long))

        self.reset()

    def reset(self) -> None:
        self.trades: List[Trade] = []
        self.entry_indices: List[int] = []

        self._pos: Optional[Position] = None
        self._cooldown: int = 0

        self._equity: float = 1.0
        self.equity_curve: List[float] = []

        self._sell_debug = {"seen": 0, "sell_actions": 0}
        self.exit_reason_counts: Dict[str, int] = {}

        self.entry_debug = {
            "checked": 0,
            "blocked_trend": 0,
            "blocked_latest_entry": 0,
            "blocked_conf": 0,
            "opened": 0,
            "conf_min": float("inf"),
            "conf_max": -float("inf"),
        }

    # ---------------------------------------------------------------------
    # Utils
    # ---------------------------------------------------------------------

    def _trend_ok(self, t: int) -> bool:
        if not self.trade_cfg.use_trend_filter:
            return True
        if self._ma_short is None or self._ma_long is None:
            return True
        if np.isnan(self._ma_short[t]) or np.isnan(self._ma_long[t]):
            return False
        return bool(self._ma_short[t] >= self._ma_long[t])

    def _buy_confidence(self, q: np.ndarray) -> float:
        method = str(self.trade_cfg.confidence_method)
        temp = float(self.trade_cfg.confidence_temp)

        if method == "margin_sigmoid":
            return margin_sigmoid_confidence(q, action=1, temp=temp)

        # default softmax
        return softmax_confidence(q, action=1, temp=temp)

    def _sell_confidence(self, q: np.ndarray) -> float:
        # Confidence that action=1 (SELL) is correct
        method = str(self.trade_cfg.confidence_method)
        temp = float(self.trade_cfg.confidence_temp)

        if method == "margin_sigmoid":
            return margin_sigmoid_confidence(q, action=1, temp=temp)
        
        # default softmax
        return softmax_confidence(q, action=1, temp=temp)

    @staticmethod
    def _sma(x: np.ndarray, window: int) -> np.ndarray:
        window = int(window)
        if window <= 1:
            return x.astype(np.float32).copy()
        out = np.full_like(x, np.nan, dtype=np.float32)
        c = np.cumsum(x, dtype=np.float64)
        c[window:] = c[window:] - c[:-window]
        out[window - 1:] = (c[window - 1:] / window).astype(np.float32)
        return out

    # -----------------------
    # segment + last_allowed (MUST MATCH SellEnv)
    # -----------------------

    def _segment_id(self, t: int) -> int:
        if self.segment_len is None:
            return 0
        return int(t // self.segment_len)

    def _segment_end(self, t: int) -> int:
        if self.segment_len is None:
            return len(self.prices) - 1
        seg = self._segment_id(t)
        end = (seg + 1) * self.segment_len - 1
        return int(min(end, len(self.prices) - 1))

    def _last_allowed_exit(self, entry_idx: int) -> int:
        """
        EXACT SellEnv._last_allowed(entry_idx):
        last_allowed = min(entry_idx + sell_horizon, segment_end(entry_idx), n-1)
        """
        seg_end = self._segment_end(entry_idx)
        return int(min(entry_idx + int(self.trade_cfg.sell_horizon), seg_end, len(self.prices) - 1))

    # ---------------------------------------------------------------------
    # SellAgent observation
    # ---------------------------------------------------------------------

    def _sell_state(self, t: int) -> np.ndarray:
        if self._pos is None:
            raise RuntimeError("_sell_state called while flat")
        if self.sell_agent is None:
            raise RuntimeError("_sell_state called but sell_agent is None")

        base = self.state[t].astype(np.float32, copy=False)

        expected_dim = int(getattr(self.sell_agent.cfg, "state_dim", base.shape[0]))
        if expected_dim == base.shape[0]:
            return base

        entry_idx = int(self._pos.entry_idx)
        entry_price = float(self._pos.entry_price)
        price_now = float(self.prices[t])

        last_allowed = int(self._last_allowed_exit(entry_idx))
        # horizon = max(1, int(self.trade_cfg.sell_horizon))

        hold = int(t - entry_idx)

        # TODO: Clean 1e-12
        unreal = (price_now - entry_price) / (entry_price + 1e-12)

        eff_h = max(1, int(last_allowed - entry_idx))
        time_frac = float(min(1.0, hold / eff_h))
        remaining = float(max(0, last_allowed - t))
        remaining_frac = float(min(1.0, remaining / eff_h))

        extra = np.array([unreal, time_frac, remaining_frac], dtype=np.float32)
        out = np.concatenate([base, extra], axis=0).astype(np.float32, copy=False)

        if out.shape[0] != expected_dim:
            raise RuntimeError(f"Sell state dim mismatch: got {out.shape[0]} expected {expected_dim}")

        return out

    # ---------------------------------------------------------------------
    # Main methods
    # ---------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        self.reset()

        n = len(self.prices)

        # Constants
        BUY_MIN_CONF = float(self.trade_cfg.buy_min_confidence)
        MIN_HOLD = int(self.trade_cfg.min_hold_bars)

        SELL_MIN_CONF = float(getattr(self.trade_cfg, "sell_min_confidence", 0.0))
        SELL_MIN_MARGIN = float(getattr(self.trade_cfg, "sell_min_margin", 0.01))

        for t in range(n):
            price = float(self.prices[t])
            s = self.state[t]

            # Cooldown tick
            if self._cooldown > 0:
                self._cooldown -= 1

            # Exits (forced limit OR sell agent)
            if self._pos is not None:
                entry_idx = int(self._pos.entry_idx)
                hold = int(t - entry_idx)

                # last_allowed computed from entry_idx (entry segment), not current t
                last_allowed = int(self._last_allowed_exit(entry_idx))

                # Forced exit: matches SellEnv "if t >= last_allowed: done"
                if t >= last_allowed:
                    seg_end_entry = int(self._segment_end(entry_idx))
                    reason = "segment_end" if last_allowed == seg_end_entry else "time"
                    self._close(t, price, forced=True, meta={"reason": reason})
                else:
                    # SellAgent decision window
                    if self.sell_agent is not None and hold >= MIN_HOLD:
                        sell_state = self._sell_state(t)
                        q_sell = np.asarray(self.sell_agent.q_values(sell_state), dtype=np.float32).reshape(-1)

                        if q_sell.shape[0] >= 2 and np.all(np.isfinite(q_sell[:2])):
                            q0, q1 = float(q_sell[0]), float(q_sell[1])
                            
                            # Computing margin
                            margin = q1 - q0
                            action = int(np.argmax(q_sell[:2]))
                            sell_conf = float(self._sell_confidence(q_sell[:2]))

                            self._sell_debug["seen"] += 1
                            if action == 1:
                                self._sell_debug["sell_actions"] += 1

                            # If agent chooses SELL and margin >= threshold and sell_conf >= threshold
                            if action == 1 and margin >= SELL_MIN_MARGIN and sell_conf >= SELL_MIN_CONF:
                                baseline = self._net_tm_at(last_allowed, entry_price=self._pos.entry_price)
                                net_now  = self._net_tm_at(t, entry_price=self._pos.entry_price)
                                delta = net_now - baseline

                                MIN_DELTA = float(getattr(self.trade_cfg, "sell_min_delta_vs_hold", 0.0))  # start at 0.0
                                if delta > MIN_DELTA:
                                    self._close(
                                        t, price, forced=False,
                                        meta={
                                            "reason": "sell_agent",
                                            "sell_conf": sell_conf,
                                            "sell_q0": q0,
                                            "sell_q1": q1,
                                            "sell_margin": margin,
                                            "sell_delta_vs_hold": float(delta),
                                            "sell_baseline_net": float(baseline),
                                            "sell_net_now": float(net_now),
                                        },
                                    )

            # Entries (only if flat)
            if self._pos is None and self._cooldown == 0:
                self.entry_debug["checked"] += 1

                if not self._trend_ok(t):
                    self.entry_debug["blocked_trend"] += 1
                else:
                    seg_end = int(self._segment_end(t))
                    latest_entry = int(seg_end - int(self.trade_cfg.sell_horizon))
                    if t > latest_entry:
                        self.entry_debug["blocked_latest_entry"] += 1
                    else:
                        q_buy = np.asarray(self.buy_agent.q_values(s), dtype=np.float32).reshape(-1)

                        if q_buy.shape[0] < 2 or not np.all(np.isfinite(q_buy[:2])):
                            self.entry_debug["blocked_conf"] += 1
                        else:
                            conf = float(self._buy_confidence(q_buy[:2]))
                            self.entry_debug["conf_min"] = min(self.entry_debug["conf_min"], conf)
                            self.entry_debug["conf_max"] = max(self.entry_debug["conf_max"], conf)

                            # Checking min_conf for Buy
                            if conf >= BUY_MIN_CONF:
                                if not check_sentiment(self.state[t], self.trade_cfg):
                                    self.entry_debug.setdefault("blocked_sentiment", 0)
                                    self.entry_debug["blocked_sentiment"] += 1

                                    # Storing a few examples
                                    self.entry_debug.setdefault("blocked_sentiment_samples", [])
                                    if len(self.entry_debug["blocked_sentiment_samples"]) < 5:
                                        sent = float(self.state[t, -2])
                                        mass = float(self.state[t, -1])
                                        self.entry_debug["blocked_sentiment_samples"].append(
                                            {"t": int(t), "sent": sent, "mass": mass, "conf": float(conf)}
                                        )
                                else:
                                    self._open(t, price, meta={"buy_conf": conf})
                                    self.entry_indices.append(int(t))
                                    self.entry_debug["opened"] += 1
                            else:
                                self.entry_debug["blocked_conf"] += 1

            # Record equity
            self.equity_curve.append(float(self._equity))

        # Forcing close at end
        if self._pos is not None:
            self._close(n - 1, float(self.prices[n - 1]), forced=True, meta={"reason": "eod"})
            if self.equity_curve:
                self.equity_curve[-1] = float(self._equity)

        return {
            "n_trades": len(self.trades),
            "final_equity": float(self._equity),
            "trades": [tr.__dict__ for tr in self.trades],
            "entry_indices": list(self.entry_indices),
            "equity_curve": list(self.equity_curve),
            "exit_reasons": dict(getattr(self, "exit_reason_counts", {})),
            "entry_debug": dict(getattr(self, "entry_debug", {})),
            "sell_debug": dict(getattr(self, "_sell_debug", {})),
        }

    # ---------------------------------------------------------------------
    # Open and close methods
    # ---------------------------------------------------------------------

    def _open(self, t: int, price: float, meta: Dict[str, Any]) -> None:
        # charge transaction cost on entry
        tc = float(self.reward_cfg.transaction_cost)
        self._equity *= (1.0 - tc)

        self._pos = Position(entry_idx=int(t), entry_price=float(price), meta=dict(meta))

    def _close(self, t: int, price: float, forced: bool, meta: Dict[str, Any]) -> None:
        if self._pos is None:
            return

        # Transaction cost
        tc = float(self.reward_cfg.transaction_cost)

        entry_idx = int(self._pos.entry_idx)
        entry_price = float(self._pos.entry_price)
        exit_price = float(price)

        gross = (exit_price - entry_price) / (entry_price + 1e-12)

        # Apply the price move only while in position (long term only)
        self._equity *= (1.0 + gross)

        # charge transaction cost on exit
        self._equity *= (1.0 - tc)

        # Net return
        net_exact = ((1.0 - tc) * (1.0 - tc) * (1.0 + gross)) - 1.0

        reason = meta.get("reason", "unknown")
        self.exit_reason_counts[reason] = self.exit_reason_counts.get(reason, 0) + 1

        self.trades.append(
            Trade(
                entry_idx=entry_idx,
                exit_idx=int(t),
                entry_price=entry_price,
                exit_price=exit_price,
                gross_return=float(gross),
                net_return=float(net_exact),
                hold_bars=int(t - entry_idx),
                forced_exit=bool(forced),
                meta={**self._pos.meta, **dict(meta)},
            )
        )

        self._pos = None
        self._cooldown = int(self.trade_cfg.cooldown_steps)

    # ---------------------------------------------------------------------
    # Entry harvesting
    # ---------------------------------------------------------------------

    def collect_entry_indices_topk(
        self,
        topk_per_segment: int = 50,
        min_gap: Optional[int] = None,
        use_confidence_score: bool = False,
    ) -> List[int]:
        """
        Harvest entry indices for SellAgent training without executing trades.

        My approach:
          - For each segment, score every eligible bar using BuyAgent.
          - Pick top-K bars by score.
          - Enforce spacing (min_gap) to avoid clustered entries.
        """

        n = len(self.prices)
        if n == 0:
            return []

        if self.segment_len is None:
            seg_count = 1
        else:
            seg_count = int(np.ceil(n / self.segment_len))

        horizon = int(self.trade_cfg.sell_horizon)
        if horizon <= 0:
            raise ValueError("sell_horizon must be > 0 for entry harvesting")

        if min_gap is None:
            min_gap = max(int(self.trade_cfg.min_hold_bars), horizon // 2, 1)

        collected: List[int] = []

        for seg_id in range(seg_count):
            seg_start = seg_id * (self.segment_len or n)
            seg_end = min((seg_id + 1) * (self.segment_len or n) - 1, n - 1)

            latest_entry = seg_end - horizon
            if latest_entry < seg_start:
                continue

            scores: List[Tuple[float, int]] = []
            for t in range(seg_start, latest_entry + 1):
                if not self._trend_ok(t):
                    continue

                s = self.state[t]
                q_buy = np.asarray(self.buy_agent.q_values(s), dtype=np.float32).reshape(-1)
                if q_buy.shape[0] < 2 or not np.all(np.isfinite(q_buy[:2])):
                    continue

                q0 = float(q_buy[0])
                q1 = float(q_buy[1])
                margin = q1 - q0

                score = float(self._buy_confidence(q_buy[:2])) if use_confidence_score else float(margin)
                scores.append((score, t))

            if not scores:
                continue

            scores.sort(key=lambda x: x[0], reverse=True)

            picked: List[int] = []
            for score, t in scores:
                if len(picked) >= int(topk_per_segment):
                    break
                if all(abs(t - p) >= int(min_gap) for p in picked):
                    picked.append(int(t))

            picked.sort()
            collected.extend(picked)

        return collected

    def _net_tm_at(self, exit_idx: int, entry_price: float) -> float:
        tc = float(self.reward_cfg.transaction_cost)
        exit_price = float(self.prices[exit_idx])
        gross = (exit_price - entry_price) / (entry_price + 1e-12)
        return ((1.0 - tc) * (1.0 - tc) * (1.0 + gross)) - 1.0

