from __future__ import annotations

import yaml
import numpy as np
import torch

# from agents.buy_agent_trainer import BuyAgentTrainer
# from agents.sell_agent_trainer import SellAgentTrainer

from config.system import TradingSystemConfig

from features.state_assembler import StateAssembler

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List

import numpy as np

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Position:
    entry_idx: int
    entry_price: float
    meta: Dict[str, Any] = field(default_factory=dict)


# @dataclass
# class TradeManagerParams:
#     # Matches your YAML:
#     # trade_manager.cooldown_steps
#     cooldown_steps: int

#     # Matches your YAML:
#     # reward.transaction_cost
#     # IMPORTANT: we keep your current semantics: round-trip cost applied ONCE at exit.
#     transaction_cost_round_trip: float

#     # Matches your YAML:
#     # trade_manager.sell_horizon
#     # Hard max holding period (time stop). If None, disabled.
#     max_hold_bars: Optional[int] = None

#     # Recommended for evaluation consistency:
#     force_close_at_end: bool = True
@dataclass
class TradeManagerParams:
    cooldown_steps: int
    transaction_cost_round_trip: float
    max_hold_bars: Optional[int] = None
    force_close_at_end: bool = True

    # NEW: do not allow selling before this many bars
    min_hold_bars: int = 1

    # Entry gating
    buy_min_confidence: Optional[float] = None
    use_trend_filter: bool = False
    ma_short: int = 10
    ma_long: int = 30
    confidence_temp: float = 0.01





class TradeManager:
    """
    Long-only, single-position trade manager (orchestrator) using two agents:
      - buy_agent: decides when to enter
      - sell_agent: decides when to exit

    Preserves your existing behavior:
      - Entry when buy_agent.select_action(state, greedy=True) == 1
      - Exit when sell_agent.select_action(state, greedy=True) == 1
      - gross_return = (exit - entry) / entry
      - net_return  = gross_return - transaction_cost_round_trip
        (round-trip cost applied once at exit, exactly like before)

    Adds:
      - sell_horizon as a hard time-stop (max_hold_bars)
      - optional force close at end-of-data (EOD) to realize PnL
      - consistent trade schema including hold_bars and forced_exit
    """

    def __init__(
        self,
        buy_agent,
        sell_agent,
        state_df,
        prices,
        config,
        params: Optional[TradeManagerParams] = None,
    ):
        self.buy_agent = buy_agent
        self.sell_agent = sell_agent
        self.state_df = state_df
        self.prices = prices
        self.config = config

        # # GA-ready: allow external overrides without mutating shared config
        # self.params = params or TradeManagerParams(
        #     cooldown_steps=int(config.trade_manager.cooldown_steps),
        #     transaction_cost_round_trip=float(config.reward.transaction_cost),
        #     max_hold_bars=int(config.trade_manager.sell_horizon)
        #     if getattr(config.trade_manager, "sell_horizon", None) is not None
        #     else None,
        #     force_close_at_end=True,
        # )
        tm_cfg = config.trade_manager

        self.params = params or TradeManagerParams(
            cooldown_steps=int(tm_cfg.cooldown_steps),
            transaction_cost_round_trip=float(config.reward.transaction_cost),
            max_hold_bars=int(tm_cfg.sell_horizon) if getattr(tm_cfg, "sell_horizon", None) is not None else None,
            force_close_at_end=True,

            # NEW
            min_hold_bars=int(getattr(tm_cfg, "min_hold_bars", 1)),

            buy_min_confidence=float(tm_cfg.buy_min_confidence) if getattr(tm_cfg, "buy_min_confidence", None) is not None else None,
            confidence_temp=float(tm_cfg.confidence_temp) if getattr(tm_cfg, "confidence_temp", None) is not None else None,
            use_trend_filter=bool(getattr(tm_cfg, "use_trend_filter", False)),
            ma_short=int(getattr(tm_cfg, "ma_short", 10)),
            ma_long=int(getattr(tm_cfg, "ma_long", 30)),
        )


        # # Precompute MAs once (fast)
        # self._ma_short = None
        # self._ma_long = None
        # if self.params.use_trend_filter:
        #     self._ma_short = self.prices.rolling(self.params.ma_short).mean()
        #     self._ma_long = self.prices.rolling(self.params.ma_long).mean()
        self._ma_short = None
        self._ma_long = None

        if self.params.use_trend_filter:
            # Support both pandas Series and numpy arrays
            if hasattr(self.prices, "rolling"):
                # pandas path
                self._ma_short = self.prices.rolling(self.params.ma_short).mean().values
                self._ma_long  = self.prices.rolling(self.params.ma_long).mean().values
            else:
                # numpy path
                self._ma_short = self._compute_sma(self.prices, self.params.ma_short)
                self._ma_long  = self._compute_sma(self.prices, self.params.ma_long)

        self.reset()

    def reset(self):
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self._position: Optional[Position] = None
        self._cooldown: int = 0
        self._equity: float = 1.0

    @property
    def in_position(self) -> bool:
        return self._position is not None

    def _get_state(self, t: int) -> np.ndarray:
        # pandas DataFrame or numpy array
        if hasattr(self.state_df, "iloc"):
            state = self.state_df.iloc[t].values.astype(np.float32)
        else:
            state = np.asarray(self.state_df[t], dtype=np.float32)

        # invariants
        assert state.ndim == 1
        assert state.shape[0] == self.buy_agent.state_dim
        return state


    # def _get_price(self, t: int) -> float:
    #     return float(self.prices.iloc[t])
    def _get_price(self, t: int) -> float:
        if hasattr(self.prices, "iloc"):
            return float(self.prices.iloc[t])
        return float(self.prices[t])


    def _open(self, t: int, price: float, meta: Optional[Dict[str, Any]] = None):
        self._position = Position(entry_idx=t, entry_price=price, meta=meta or {})

    def _close(
        self,
        t: int,
        price: float,
        forced: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Trade]:
        pos = self._position
        if pos is None:
            return None

        hold_bars = t - pos.entry_idx

        gross_return = (price - pos.entry_price) / pos.entry_price

        # EXACTLY your previous semantics:
        net_return = gross_return - self.params.transaction_cost_round_trip

        self._equity *= (1.0 + net_return)

        trade = Trade(
            entry_idx=pos.entry_idx,
            exit_idx=t,
            entry_price=pos.entry_price,
            exit_price=price,
            gross_return=float(gross_return),
            net_return=float(net_return),
            hold_bars=int(hold_bars),
            forced_exit=bool(forced),
            meta={**pos.meta, **(meta or {})},
        )
        self.trades.append(trade)

        self._position = None
        self._cooldown = self.params.cooldown_steps
        return trade

    def _time_stop_hit(self, t: int) -> bool:
        if self._position is None:
            return False
        if self.params.max_hold_bars is None:
            return False
        return (t - self._position.entry_idx) >= self.params.max_hold_bars

    def run(self) -> Dict[str, Any]:
        self.reset()

        n = len(self.prices)

        for t in range(n):
            state = self._get_state(t)
            price = self._get_price(t)

            # 0) Forced time-stop (sell_horizon / max_hold_bars)
            if self.in_position and self._time_stop_hit(t):
                self._close(t, price, forced=True, meta={"reason": "time"})
                # after a close we still tick cooldown below and record equity
                # and we DO NOT allow re-entry on the same bar (cooldown > 0 anyway)

            # 1) ENTRY
            if (not self.in_position) and self._cooldown == 0:
                should_enter, meta = self._should_enter(t, state)
                if should_enter:
                    self._open(t, price, meta=meta)

            # 2) EXIT (agent-driven)
            elif self.in_position:
                hold_bars = t - self._position.entry_idx

                # Minimum holding period gate (but forced exits above still apply)
                if hold_bars >= self.params.min_hold_bars:
                    action, conf = self._sell_action_and_conf(state)
                    if action == 1:
                        self._close(
                            t,
                            price,
                            forced=False,
                            meta={
                                "reason": "agent",
                                "sell_confidence": float(conf) if conf is not None else None,
                            },
                        )

            # 3) COOLDOWN TICK
            if self._cooldown > 0:
                self._cooldown -= 1

            self.equity_curve.append(self._equity)

        # 4) End-of-data forced close (recommended for evaluation consistency)
        if self.in_position and self.params.force_close_at_end:
            last_t = n - 1
            last_price = self._get_price(last_t)
            self._close(last_t, last_price, forced=True, meta={"reason": "eod"})
            if self.equity_curve:
                self.equity_curve[-1] = self._equity

        return {
            "final_equity": float(self._equity),
            "equity_curve": self.equity_curve,
            "trades": [tr.to_dict() for tr in self.trades],
        }

    
    # def _trend_ok(self, t: int) -> bool:
    #     if not self.params.use_trend_filter:
    #         return True
    #     if self._ma_short is None or self._ma_long is None:
    #         return True  # trend filter enabled but MA not available -> fail open or fail closed
    #     ms = float(self._ma_short.iloc[t])
    #     ml = float(self._ma_long.iloc[t])
    #     if np.isnan(ms) or np.isnan(ml):
    #         return False  # conservative early-window behavior
    #     return ms > ml
    def _trend_ok(self, t: int) -> bool:
        if not self.params.use_trend_filter:
            return True

        ms = self._ma_short[t]
        ml = self._ma_long[t]

        if np.isnan(ms) or np.isnan(ml):
            return False

        return ms > ml


    # def _buy_action_and_conf(self, state: np.ndarray) -> tuple[int, Optional[float]]:
    #     """
    #     Prefer agent.act_with_confidence if available.
    #     Falls back to select_action with confidence=None.
    #     """
    #     if hasattr(self.buy_agent, "act_with_confidence"):
    #         a, c = self.buy_agent.act_with_confidence(state)
    #         return int(a), float(c)
    #     a = self.buy_agent.select_action(state, greedy=True)
    #     return int(a), None
    def _buy_action_and_conf(self, state: np.ndarray) -> tuple[int, Optional[float], Optional[float]]:
        if hasattr(self.buy_agent, "q_values"):
            # q = self.buy_agent.q_values(state)
            # action = int(np.argmax(q))

            # buy_idx = 1
            # q_buy = float(q[buy_idx])
            # q_other = float(np.max(np.delete(q, buy_idx)))
            # raw_margin = q_buy - q_other

            # margin = raw_margin / max(1e-8, self.params.confidence_temp)
            # conf = 1.0 / (1.0 + np.exp(-margin))
            # return action, float(conf), float(raw_margin)
            q = self.buy_agent.q_values(state)
            q = np.asarray(q, dtype=np.float32)

            action = int(np.argmax(q))
            q_best = float(q[action])

            # best alternative
            q_alt = float(np.max(np.delete(q, action))) if q.shape[0] > 1 else q_best
            raw_margin = q_best - q_alt  # >= 0 by construction (ties -> 0)

            scaled = raw_margin / max(1e-8, self.params.confidence_temp)
            conf = 1.0 / (1.0 + np.exp(-scaled))

            return action, float(conf), float(raw_margin)


        a = self.buy_agent.select_action(state, greedy=True)
        return int(a), None, None



    def _should_enter(self, t: int, state: np.ndarray) -> tuple[bool, Dict[str, Any]]:
        meta: Dict[str, Any] = {}

        action, conf, raw_margin = self._buy_action_and_conf(state)  # <-- 3 values
        meta["buy_action"] = action
        meta["buy_confidence"] = conf
        meta["buy_margin_raw"] = raw_margin

        if action != 1:
            return False, meta

        # Confidence gate
        if self.params.buy_min_confidence is not None and conf is not None:
            if conf < self.params.buy_min_confidence:
                return False, meta

        # Trend gate
        trend_ok = self._trend_ok(t)
        meta["trend_ok"] = trend_ok
        if self.params.use_trend_filter and not trend_ok:
            return False, meta

        return True, meta
    
    def _compute_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """
        Simple moving average for numpy arrays.
        Returns array of same length with np.nan for unavailable positions.
        """
        if window <= 1:
            return prices.astype(np.float32)

        prices = prices.astype(np.float32)
        out = np.full_like(prices, np.nan, dtype=np.float32)

        csum = np.cumsum(prices, dtype=np.float32)
        csum[window:] = csum[window:] - csum[:-window]
        out[window - 1:] = csum[window - 1:] / float(window)

        return out
    
    def _sell_action_and_conf(self, state: np.ndarray) -> tuple[int, float | None]:
        """
        Returns (action, confidence) for SELL agent.
        Confidence is margin-based on chosen action, same as buy.
        """
        # If agent exposes q_values(), compute margin confidence
        if hasattr(self.sell_agent, "q_values"):
            q = np.asarray(self.sell_agent.q_values(state), dtype=np.float32)
            action = int(np.argmax(q))
            q_best = float(q[action])
            q_alt = float(np.max(np.delete(q, action))) if q.shape[0] > 1 else q_best
            margin = q_best - q_alt
            scaled = margin / max(1e-8, self.params.confidence_temp)
            # conf = 1.0 / (1.0 + np.exp(-scaled))
            conf = self._confidence_from_q(q_values, action)
            return action, float(conf)

        # Fallback: no confidence available
        action = int(self.sell_agent.select_action(state, greedy=True))
        return action, None
    
    def _confidence_from_q(self, q_values, action):
        q = np.array(q_values, dtype=np.float32)
        temp = max(1e-6, self.params.confidence_temp)

        if getattr(self.params, "confidence_method", "softmax") == "margin_sigmoid":
            q_best = q[action]
            q_alt = np.max(np.delete(q, action))
            margin = q_best - q_alt
            return float(1.0 / (1.0 + np.exp(-margin / temp)))

        # DEFAULT: softmax
        exp_q = np.exp(q / temp)
        probs = exp_q / np.sum(exp_q)
        return float(probs[action])





