import gym
import numpy as np
from typing import Dict, Any

from config.system import TradingSystemConfig


class SellEnv(gym.Env):
    """
    Sell-only trading environment.

    The agent always starts LONG and must decide when to SELL.

    Actions:
        0 = HOLD
        1 = SELL (close position)

    Episode:
        - Ends on SELL
        - Or forced close at horizon
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        state_df: np.ndarray,
        prices: np.ndarray,
        entry_indices: np.ndarray,
        config: TradingSystemConfig,
    ):
        super().__init__()

        # Safety check
        if entry_indices is None or len(entry_indices) == 0:
            raise ValueError(
                "SellEnv received empty entry_indices. "
                "SellAgent cannot train without BUY entry points."
            )


        # -----------------------
        # Data
        # -----------------------
        self.state_df = state_df
        self.prices = prices
        self.entry_indices = entry_indices
        self.n_steps = len(prices)

        # -----------------------
        # Config references
        # -----------------------
        self.reward_cfg = config.reward
        self.trade_cfg = config.trade_manager

        # -----------------------
        # Action / Observation
        # -----------------------
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_df.shape[1],),
            dtype=np.float32,
        )

        # -----------------------
        # Internal state
        # -----------------------
        self.reset()

    # --------------------------------------------------
    # Gym API
    # --------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Start a new episode by sampling a valid entry index.
        """
        self.entry_t = int(np.random.choice(self.entry_indices))
        self.t = self.entry_t

        self.entry_price = self.prices[self.entry_t]
        self.done = False

        return self._get_obs()

    def step(self, action: int):
        reward = 0.0
        info: Dict[str, Any] = {}

        price = self.prices[self.t]
        hold_bars = self.t - self.entry_t

        # -----------------------
        # SELL action
        # -----------------------
        if action == 1:
            reward = self._close_position(price, forced=False)
            self.done = True
            info["forced_exit"] = False

        # -----------------------
        # Horizon-based forced close
        # -----------------------
        elif hold_bars >= self.trade_cfg.sell_horizon:
            reward = self._close_position(price, forced=True)
            self.done = True
            info["forced_exit"] = True

        # -----------------------
        # Advance time
        # -----------------------
        if not self.done:
            self.t += 1
            if self.t >= self.n_steps - 1:
                reward = self._close_position(price, forced=True)
                self.done = True
                info["forced_exit"] = True

        return self._get_obs(), reward, self.done, info

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _close_position(self, exit_price: float, forced: bool) -> float:
        gross_return = (exit_price - self.entry_price) / self.entry_price
        net_return = gross_return - self.reward_cfg.transaction_cost

        reward = self._compute_reward(net_return)

        return reward

    def _compute_reward(self, net_return: float) -> float:
        """
        Centralised reward shaping.
        """
        reward = net_return

        # Drawdown proxy penalty
        reward -= self.reward_cfg.lambda_dd * max(0.0, -net_return)

        # Volatility proxy penalty
        reward -= self.reward_cfg.lambda_vol * (net_return ** 2)

        return reward

    def _get_obs(self) -> np.ndarray:
        return self.state_df[self.t].astype(np.float32)
