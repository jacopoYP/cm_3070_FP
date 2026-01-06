import gym
import numpy as np
from typing import Dict, Any

from config.system import TradingSystemConfig


class SellEnv(gym.Env):
    """
    Sell-only environment.

    Episode starts LONG at a sampled BUY entry index.
    Agent chooses when to SELL.

    Actions:
        0 = HOLD
        1 = SELL

    Done:
        - SELL
        - OR forced close when hold_bars >= sell_horizon
        - OR forced close at end-of-data
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        state_df,
        prices,
        entry_indices: np.ndarray,
        config: TradingSystemConfig,
    ):
        super().__init__()

        if entry_indices is None or len(entry_indices) == 0:
            raise ValueError(
                "SellEnv received empty entry_indices. "
                "SellAgent cannot train without BUY entry points."
            )

        self.state_df = state_df
        self.prices = prices
        self.entry_indices = np.asarray(entry_indices, dtype=np.int64)

        self.n_steps = len(prices)

        self.reward_cfg = config.reward
        self.trade_cfg = config.trade_manager

        n_features = state_df.shape[1] if hasattr(state_df, "shape") else len(self._get_row(0))

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32,
        )

        self.reset()

    # -----------------------
    # Gym API
    # -----------------------

    def reset(self) -> np.ndarray:
        self.entry_t = int(np.random.choice(self.entry_indices))

        # Clamp entry so we can always progress at least one step safely
        self.entry_t = min(self.entry_t, self.n_steps - 2)

        self.t = self.entry_t
        self.entry_price = float(self._get_price(self.entry_t))
        self.done = False
        return self._get_obs()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Step called after done=True. Call reset().")

        info: Dict[str, Any] = {}

        price_t = float(self._get_price(self.t))
        # hold_bars = int(self.t - self.entry_t)
        hold_bars = self.t - self.entry_t

        min_hold = int(getattr(self.trade_cfg, "min_hold_bars", 1))
        if hold_bars < min_hold:
            action = 0


        # 1) SELL (signal exit)
        if action == 1:
            reward = self._close_position(exit_price=price_t)
            self.done = True
            info.update({
                "forced_exit": False,
                "reason": "signal",
                "entry_t": self.entry_t,
                "exit_t": self.t,
                "hold_bars": hold_bars,
            })
            return self._get_obs(), float(reward), self.done, info

        # 2) Time-stop (sell_horizon)
        if hold_bars >= int(self.trade_cfg.sell_horizon):
            reward = self._close_position(exit_price=price_t)
            self.done = True
            info.update({
                "forced_exit": True,
                "reason": "time",
                "entry_t": self.entry_t,
                "exit_t": self.t,
                "hold_bars": hold_bars,
            })
            return self._get_obs(), float(reward), self.done, info

        # 3) Advance time
        self.t += 1

        # End-of-data forced close ON the last bar with correct price
        if self.t >= self.n_steps - 1:
            price_last = float(self._get_price(self.t))
            reward = self._close_position(exit_price=price_last)
            self.done = True
            info.update({
                "forced_exit": True,
                "reason": "eod",
                "entry_t": self.entry_t,
                "exit_t": self.t,
                "hold_bars": int(self.t - self.entry_t),
            })
            return self._get_obs(), float(reward), self.done, info

        # 4) Normal HOLD step (not done)
        reward = 0.0
        info.update({
            "forced_exit": False,
            "reason": "hold",
            "entry_t": self.entry_t,
            "exit_t": None,
            "hold_bars": int(self.t - self.entry_t),
        })
        return self._get_obs(), float(reward), self.done, info

    # -----------------------
    # Helpers
    # -----------------------

    def _close_position(self, exit_price: float) -> float:
        gross_return = (exit_price - self.entry_price) / self.entry_price
        net_return = gross_return - float(self.reward_cfg.transaction_cost)
        return float(self._compute_reward(float(net_return)))

    def _compute_reward(self, net_return: float) -> float:
        reward = net_return
        reward -= float(self.reward_cfg.lambda_dd) * max(0.0, -net_return)
        reward -= float(self.reward_cfg.lambda_vol) * (net_return ** 2)
        return float(reward)

    def _get_row(self, t: int):
        # pandas DataFrame or numpy array
        return self.state_df.iloc[t].values if hasattr(self.state_df, "iloc") else self.state_df[t]

    def _get_obs(self) -> np.ndarray:
        return np.asarray(self._get_row(self.t), dtype=np.float32)

    def _get_price(self, t: int) -> float:
        # pandas Series or numpy array
        return float(self.prices.iloc[t]) if hasattr(self.prices, "iloc") else float(self.prices[t])
