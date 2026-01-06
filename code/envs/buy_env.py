# import gym
# import numpy as np
# from typing import Dict, Any
# from config.system import TradingSystemConfig


# class BuyEnv(gym.Env):
#     """
#     Buy-only trading environment.
#     Actions:
#         0 = HOLD
#         1 = BUY
#     """

#     metadata = {"render.modes": []}

#     def __init__(
#         self,
#         features: np.ndarray,
#         prices: np.ndarray,
#         config: TradingSystemConfig,
#     ):
#         super().__init__()

#         self.features = features
#         self.prices = prices
#         self.n_steps = len(prices)

#         self.reward_cfg = config.reward
#         self.trade_cfg = config.trade_manager

#         self.action_space = gym.spaces.Discrete(2)
#         self.observation_space = gym.spaces.Box(
#             low=-np.inf,
#             high=np.inf,
#             shape=(features.shape[1],),
#             dtype=np.float32,
#         )

#         self.reset()

#     def reset(self) -> np.ndarray:
#         self.t = 0
#         self.position_open = False
#         self.entry_price = None
#         self.entry_t = None
#         self.cooldown = 0
#         self.done = False
#         return self._get_obs()

#     def step(self, action: int):
#         reward = 0.0
#         info: Dict[str, Any] = {}

#         price = self.prices[self.t]

#         if self.cooldown > 0:
#             action = 0
#             self.cooldown -= 1

#         if action == 1 and not self.position_open:
#             self.position_open = True
#             self.entry_price = price
#             self.entry_t = self.t
#             info["opened"] = True

#         if self.position_open:
#             hold_bars = self.t - self.entry_t
#             if hold_bars >= self.trade_cfg.sell_horizon:
#                 reward, trade_info = self._close_position(price, forced=True)
#                 info.update(trade_info)

#         self.t += 1
#         if self.t >= self.n_steps - 1:
#             self.done = True
#             if self.position_open:
#                 r, trade_info = self._close_position(price, forced=True)
#                 reward += r
#                 info.update(trade_info)

#         return self._get_obs(), reward, self.done, info

#     def _close_position(self, exit_price: float, forced: bool):
#         gross_return = (exit_price - self.entry_price) / self.entry_price
#         net_return = gross_return - self.reward_cfg.transaction_cost

#         reward = self._compute_reward(net_return)

#         self.position_open = False
#         self.entry_price = None
#         self.entry_t = None
#         self.cooldown = self.trade_cfg.cooldown_steps

#         return reward, {
#             "gross_return": gross_return,
#             "net_return": net_return,
#             "forced_exit": forced,
#         }

#     def _compute_reward(self, net_return: float) -> float:
#         reward = net_return
#         reward -= self.reward_cfg.lambda_dd * max(0.0, -net_return)
#         reward -= self.reward_cfg.lambda_vol * abs(net_return)
#         return reward

#     def _get_obs(self) -> np.ndarray:
#         return self.features[self.t].astype(np.float32)
import gym
import numpy as np
from typing import Dict, Any
from config.system import TradingSystemConfig


class BuyEnv(gym.Env):
    """
    Buy-only trading environment (multi-trade per episode).

    Actions:
        0 = HOLD
        1 = BUY

    Mechanics:
        - BUY opens a long position if flat and not in cooldown
        - Position closes only via:
            * time-stop: hold_bars >= sell_horizon
            * end-of-data (EOD)
        - After a close, cooldown applies
        - Episode ends at end-of-data
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
        if self.done:
            raise RuntimeError("Step called after done=True. Call reset().")

        reward = 0.0
        info: Dict[str, Any] = {}

        price_t = float(self._get_price(self.t))

        # Cooldown overrides action
        if self.cooldown > 0:
            action = 0
            self.cooldown -= 1

        # 1) Entry
        if action == 1 and not self.position_open:
            self.position_open = True
            self.entry_price = price_t
            self.entry_t = self.t
            info["opened"] = True
            info["entry_t"] = self.entry_t
            info["entry_price"] = self.entry_price

        # 2) Time-stop close (sell_horizon)
        if self.position_open:
            hold_bars = int(self.t - self.entry_t)
            if hold_bars >= int(self.trade_cfg.sell_horizon):
                r, trade_info = self._close_position(exit_price=price_t, forced=True, reason="time")
                reward += r
                info.update(trade_info)

        # 3) Advance time
        self.t += 1

        # 4) End-of-data termination (close at LAST bar with correct price)
        if self.t >= self.n_steps - 1:
            self.done = True
            if self.position_open:
                price_last = float(self._get_price(self.t))  # ✅ correct last-bar price
                r, trade_info = self._close_position(exit_price=price_last, forced=True, reason="eod")
                reward += r
                info.update(trade_info)

        return self._get_obs(), float(reward), self.done, info

    def _close_position(self, exit_price: float, forced: bool, reason: str):
        gross_return = (exit_price - float(self.entry_price)) / float(self.entry_price)
        net_return = gross_return - float(self.reward_cfg.transaction_cost)

        reward = self._compute_reward(float(net_return))

        entry_t = int(self.entry_t)
        entry_price = float(self.entry_price)

        self.position_open = False
        self.entry_price = None
        self.entry_t = None
        self.cooldown = int(self.trade_cfg.cooldown_steps)

        return float(reward), {
            "closed": True,
            "reason": reason,
            "forced_exit": bool(forced),
            "entry_t": entry_t,
            "exit_t": int(self.t),
            "hold_bars": int(self.t - entry_t),
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "gross_return": float(gross_return),
            "net_return": float(net_return),
        }

    def _compute_reward(self, net_return: float) -> float:
        reward = net_return
        reward -= float(self.reward_cfg.lambda_dd) * max(0.0, -net_return)
        reward -= float(self.reward_cfg.lambda_vol) * (net_return ** 2)  # ✅ align with SellEnv
        return float(reward)

    def _get_obs(self) -> np.ndarray:
        t = min(self.t, self.n_steps - 1)
        return self.features[t].astype(np.float32)

    def _get_price(self, t: int) -> float:
        return float(self.prices.iloc[t]) if hasattr(self.prices, "iloc") else float(self.prices[t])
