# envs/buy_env.py

import gym
import numpy as np
from gym import spaces


class BuyEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, state_window_df, price_series, horizon=5, transaction_cost=0.001):
        super().__init__()

        self.states = state_window_df.values.astype(np.float32)
        self.prices = price_series.values.astype(np.float32)

        self.horizon = horizon
        self.cost = transaction_cost
        self.idx = 0

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.states.shape[1],), dtype=np.float32
        )

    def reset(self):
        self.idx = 0
        return self.states[self.idx]

    def step(self, action):
        done = False
        reward = 0.0

        if action == 1:  # BUY
            entry = self.prices[self.idx]
            exit_idx = min(self.idx + self.horizon, len(self.prices) - 1)
            exit_price = self.prices[exit_idx]
            reward = (exit_price - entry) / entry - self.cost
            #  TODO: Change for PROD release to keep on training
            # done = False
            done = True

        self.idx += 1
        if self.idx >= len(self.states) - 1:
            done = True

        next_state = self.states[self.idx]

        return next_state, float(reward), done, {"t": self.idx, "price": float(self.prices[self.idx])}

    def render(self):
        print(f"t={self.idx}, price={self.prices[self.idx]}")
