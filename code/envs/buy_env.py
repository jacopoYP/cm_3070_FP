# import gym
# import numpy as np
# from gym import spaces

# from config.system import TradingSystemConfig


# class BuyEnv(gym.Env):
#     """
#     Production BUY environment.

#     Actions:
#         0 = HOLD
#         1 = BUY

#     Behaviour:
#     - When flat and agent selects BUY (1):
#         -> open a long position at current price.
#     - When already long and agent selects BUY:
#         -> treated as HOLD (no scaling/leverage for now).

#     - Position is automatically closed when:
#         -> holding_steps >= horizon, or
#         -> we reach the end of the dataset.

#     Reward:
#     - 0 (or tiny penalties) on normal HOLD steps.
#     - On CLOSE (normal or forced at episode end):
#         reward = gross_return - transaction_cost
#                  - lambda_dd * |max_drawdown|
#                  - lambda_vol * volatility
#                  - lambda_tf * num_trades

#     Extra:
#     - Cooldown period after each close, during which agent is forced to HOLD.
#     - NaN-safe for all metrics.
#     """

#     # metadata = {"render.modes": ["human"]}
#     metadata = {"render.modes": []}

#     # def __init__(
#     #     self,
#     #     state_window_df,
#     #     price_series,
#     #     horizon: int = 10,
#     #     transaction_cost: float = 0.001,
#     #     # Tuned penalties (milder than before)
#     #     lambda_dd: float = 0.1,      # drawdown penalty scale
#     #     lambda_vol: float = 0.02,    # volatility penalty scale
#     #     lambda_tf: float = 0.0005,   # trade frequency penalty scale
#     #     # Per-step penalties (optional, can be 0)
#     #     hold_penalty_flat: float = 0.0,
#     #     hold_penalty_long: float = 0.0,
#     #     # Episode control
#     #     max_episode_steps: int | None = None,
#     #     # Cooldown after closing a trade
#     #     cooldown_period: int = 5,
#     # ):
#     def __init__(
#         self,
#         state_df: np.ndarray,
#         prices: np.ndarray,
#         config: TradingSystemConfig,
#     ):
#         super().__init__()

#         # self.state_df = state_window_df
#         # self.price_series = price_series
#         # self.horizon = horizon
#         # self.transaction_cost = transaction_cost
#         # self.lambda_dd = lambda_dd
#         # self.lambda_vol = lambda_vol
#         # self.lambda_tf = lambda_tf
#         # self.hold_penalty_flat = hold_penalty_flat
#         # self.hold_penalty_long = hold_penalty_long

#         # self.num_states = state_window_df.shape[0]
#         # self.num_features = state_window_df.shape[1]

#         # # Gym spaces
#         # self.action_space = spaces.Discrete(2)  # 0 = HOLD, 1 = BUY
#         # self.observation_space = spaces.Box(
#         #     low=-np.inf,
#         #     high=np.inf,
#         #     shape=(self.num_features,),
#         #     dtype=np.float32,
#         # )

#         # # Episode internals
#         # self.idx: int = 0
#         # self.position: int = 0  # 0 = flat, 1 = long
#         # self.entry_price: float | None = None
#         # self.entry_idx: int | None = None
#         # self.num_trades: int = 0
#         # self.episode_steps: int = 0
#         # self.max_episode_steps = max_episode_steps

#         # # Cooldown state
#         # self.cooldown_period = cooldown_period
#         # self.cooldown_steps: int = 0
#         # -----------------------
#         # Data
#         # -----------------------
#         self.state_df = state_df
#         self.prices = prices
#         self.n_steps = len(prices)

#         # -----------------------
#         # Config references
#         # -----------------------
#         self.reward_cfg = config.reward
#         self.trade_cfg = config.trade_manager

#         # -----------------------
#         # Action / Observation
#         # -----------------------
#         self.action_space = gym.spaces.Discrete(2)
#         self.observation_space = gym.spaces.Box(
#             low=-np.inf,
#             high=np.inf,
#             shape=(state_df.shape[1],),
#             dtype=np.float32,
#         )

#         # -----------------------
#         # Internal state
#         # -----------------------
#         self.reset()

#     # ------------------------------------------------------------------ #
#     # Core API
#     # ------------------------------------------------------------------ #

#     def reset(self):
#         self.idx = 0
#         self.position = 0
#         self.entry_price = None
#         self.entry_idx = None
#         self.num_trades = 0
#         self.episode_steps = 0
#         self.cooldown_steps = 0

#         return self._get_state()

#     def _get_state(self) -> np.ndarray:
#         return self.state_df.iloc[self.idx].values.astype(np.float32)

#     def step(self, action: int):
#         """
#         Step the environment by one time step.

#         1) Apply cooldown override (forces HOLD).
#         2) Apply action given current position.
#         3) Check close conditions, compute trade reward if closing.
#         4) Advance time, maybe force close at the end.
#         """
#         assert self.action_space.contains(action), f"Invalid action {action}"

#         done = False
#         info = {}
#         reward = 0.0

#         # ------------------------------------------------------------------
#         # 0) Cooldown override: force HOLD if cooling down
#         # ------------------------------------------------------------------
#         if self.cooldown_steps > 0:
#             action = 0  # HOLD
#             self.cooldown_steps -= 1

#         current_price = self.price_series.iloc[self.idx]

#         # ------------------------------------------------------------------
#         # 1) Apply Action
#         # ------------------------------------------------------------------
#         if self.position == 0:
#             # FLAT
#             if action == 1:
#                 # Open new long
#                 self.position = 1
#                 self.entry_price = float(current_price)
#                 self.entry_idx = self.idx
#                 self.num_trades += 1
#                 # We apply transaction cost at close, not here.
#             else:
#                 # HOLD while flat
#                 reward -= self.hold_penalty_flat
#         else:
#             # LONG (BUY treated as HOLD for now)
#             reward -= self.hold_penalty_long

#         # ------------------------------------------------------------------
#         # 2) Check Close Conditions (normal close)
#         # ------------------------------------------------------------------
#         close_position = False

#         if self.position == 1 and self.entry_idx is not None:
#             holding_steps = self.idx - self.entry_idx
#             if holding_steps >= self.horizon or self.idx >= self.num_states - 1:
#                 close_position = True

#         if close_position and self.position == 1 and self.entry_idx is not None:
#             trade_reward, trade_info = self._compute_trade_reward(
#                 entry_idx=self.entry_idx,
#                 exit_idx=self.idx,
#             )
#             reward += trade_reward

#             # Reset position
#             self.position = 0
#             self.entry_price = None
#             self.entry_idx = None

#             # Start cooldown after close
#             self.cooldown_steps = self.cooldown_period

#             # Merge trade info into info dict
#             info.update(trade_info)

#         # ------------------------------------------------------------------
#         # 3) Advance Time
#         # ------------------------------------------------------------------
#         self.episode_steps += 1
#         self.idx += 1

#         # Episode termination conditions
#         if self.idx >= self.num_states - 1:
#             done = True
#         if self.max_episode_steps is not None and self.episode_steps >= self.max_episode_steps:
#             done = True

#         # ------------------------------------------------------------------
#         # 4) Force close if episode ended while still LONG
#         # ------------------------------------------------------------------
#         if done and self.position == 1 and self.entry_idx is not None:
#             # Force close at final valid index
#             final_idx = min(self.idx, self.num_states - 1)
#             closing_reward, closing_info = self._compute_trade_reward(
#                 entry_idx=self.entry_idx,
#                 exit_idx=final_idx,
#                 forced=True,
#             )
#             reward += closing_reward

#             # Clear position
#             self.position = 0
#             self.entry_price = None
#             self.entry_idx = None

#             info.update(closing_info)

#         # ------------------------------------------------------------------
#         # 5) Next State
#         # ------------------------------------------------------------------
#         if not done:
#             next_state = self._get_state()
#         else:
#             next_state = np.zeros(self.num_features, dtype=np.float32)

#         return next_state, float(reward), done, info

#     # ------------------------------------------------------------------ #
#     # Reward computation (NaN-safe)
#     # ------------------------------------------------------------------ #

#     def _compute_trade_reward(
#         self,
#         entry_idx: int,
#         exit_idx: int,
#         forced: bool = False,
#     ) -> tuple[float, dict]:
#         """
#         Compute risk-adjusted reward for a trade between entry_idx and exit_idx.
#         Fully NaN-safe.
#         """
#         # Clamp indices
#         entry_idx = max(0, min(entry_idx, self.num_states - 1))
#         exit_idx = max(0, min(exit_idx, self.num_states - 1))

#         entry_price = float(self.price_series.iloc[entry_idx])
#         exit_price = float(self.price_series.iloc[exit_idx])

#         # Gross return
#         if entry_price == 0 or np.isnan(entry_price) or np.isnan(exit_price):
#             gross_return = 0.0
#         else:
#             gross_return = (exit_price - entry_price) / entry_price

#         # Price window for risk metrics
#         price_window = self.price_series.iloc[entry_idx : exit_idx + 1].astype(float)

#         if len(price_window) > 1:
#             peak = float(price_window.max())
#             trough = float(price_window.min())

#             if peak == 0 or np.isnan(peak) or np.isnan(trough):
#                 max_drawdown = 0.0
#             else:
#                 max_drawdown = (trough - peak) / peak

#             returns = price_window.pct_change().dropna()
#             if len(returns) > 1:
#                 volatility = float(returns.std())
#             else:
#                 volatility = 0.0
#         else:
#             max_drawdown = 0.0
#             volatility = 0.0

#         # Penalties
#         drawdown_penalty = self.lambda_dd * abs(max_drawdown)
#         volatility_penalty = self.lambda_vol * abs(volatility)
#         frequency_penalty = self.lambda_tf * self.num_trades

#         trade_reward = (
#             gross_return
#             - self.transaction_cost
#             - drawdown_penalty
#             - volatility_penalty
#             - frequency_penalty
#         )

#         info = {
#             "trade_reward": trade_reward,
#             "gross_return": gross_return,
#             "max_drawdown": max_drawdown,
#             "volatility": volatility,
#             "forced_close": forced,
#         }

#         return float(trade_reward), info

#     def render(self, mode="human"):
#         pass

import gym
import numpy as np
from typing import Dict, Any
from config.system import TradingSystemConfig


class BuyEnv(gym.Env):
    """
    Buy-only trading environment.
    Actions:
        0 = HOLD
        1 = BUY
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        config: TradingSystemConfig,
    ):
        super().__init__()

        self.features = features
        self.prices = prices
        self.n_steps = len(prices)

        self.reward_cfg = config.reward
        self.trade_cfg = config.trade_manager

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(features.shape[1],),
            dtype=np.float32,
        )

        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.position_open = False
        self.entry_price = None
        self.entry_t = None
        self.cooldown = 0
        self.done = False
        return self._get_obs()

    def step(self, action: int):
        reward = 0.0
        info: Dict[str, Any] = {}

        price = self.prices[self.t]

        if self.cooldown > 0:
            action = 0
            self.cooldown -= 1

        if action == 1 and not self.position_open:
            self.position_open = True
            self.entry_price = price
            self.entry_t = self.t
            info["opened"] = True

        if self.position_open:
            hold_bars = self.t - self.entry_t
            if hold_bars >= self.trade_cfg.sell_horizon:
                reward, trade_info = self._close_position(price, forced=True)
                info.update(trade_info)

        self.t += 1
        if self.t >= self.n_steps - 1:
            self.done = True
            if self.position_open:
                r, trade_info = self._close_position(price, forced=True)
                reward += r
                info.update(trade_info)

        return self._get_obs(), reward, self.done, info

    def _close_position(self, exit_price: float, forced: bool):
        gross_return = (exit_price - self.entry_price) / self.entry_price
        net_return = gross_return - self.reward_cfg.transaction_cost

        reward = self._compute_reward(net_return)

        self.position_open = False
        self.entry_price = None
        self.entry_t = None
        self.cooldown = self.trade_cfg.cooldown_steps

        return reward, {
            "gross_return": gross_return,
            "net_return": net_return,
            "forced_exit": forced,
        }

    def _compute_reward(self, net_return: float) -> float:
        reward = net_return
        reward -= self.reward_cfg.lambda_dd * max(0.0, -net_return)
        reward -= self.reward_cfg.lambda_vol * abs(net_return)
        return reward

    def _get_obs(self) -> np.ndarray:
        return self.features[self.t].astype(np.float32)
