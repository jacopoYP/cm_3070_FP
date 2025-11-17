# envs/buy_env.py

import gym
import numpy as np
from gym import spaces


class BuyEnv(gym.Env):
    """
    Simple environment for training the Buy agent.

    - State: flattened rolling window of features (from StateAssembler)
    - Actions: 0 = HOLD, 1 = BUY
    - Reward (if BUY): K-day forward return minus transaction cost
    - Episode ends as soon as we BUY or we run out of data
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        state_window_df,
        price_series,
        horizon=5,
        transaction_cost=0.001,
    ):
        super().__init__()

        assert state_window_df.shape[0] == price_series.shape[0], \
    f"States and prices length mismatch: {state_window_df.shape[0]} vs {price_series.shape[0]}"


        print("---- ENV DEBUG ----")
        print("states shape:", state_window_df.values.shape, "dtype:", state_window_df.values.dtype)
        print("prices shape:", price_series.values.shape, "dtype:", price_series.values.dtype)
        print("nan in states:", np.isnan(state_window_df.values).sum())
        print("nan in prices:", np.isnan(price_series.values).sum())
        print("--------------------")

        # Store data as numpy arrays for speed
        self.states = state_window_df.values          # shape: (T, window * features)
        self.prices = price_series.values             # shape: (T,)
        self.horizon = horizon
        self.cost = transaction_cost

        self.idx = 0  # current time index

        # Actions: HOLD or BUY
        self.action_space = spaces.Discrete(2)

        # Observations: 1D state vector
        state_dim = self.states.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )

    def reset(self):
        """
        Reset episode to the beginning of the time series.
        """
        self.idx = 0

        # return first state
        return self.states[self.idx].astype(np.float32)

    def step(self, action):
        """
        One step in the environment.
        """
        done = False
        reward = 0.0

        # If agent decides to BUY
        if action == 1:
            # Current price
            entry_price = self.prices[self.idx]

            # Compute index of horizon
            future_idx = min(self.idx + self.horizon, len(self.prices) - 1)
            exit_price = self.prices[future_idx]

            # Simple K-day forward return
            gross_return = (exit_price - entry_price) / entry_price

            # Subtract transaction cost
            reward = gross_return - self.cost

            done = True  # we stop after a buy decision

        # If HOLD, no immediate reward, just move forward
        self.idx += 1
        if self.idx >= len(self.states) - 1:
            done = True

        next_state = self.states[self.idx].astype(np.float32)

        info = {
            "t": self.idx,
            "price": float(self.prices[self.idx]),
        }

        return next_state, float(reward), done, info

    def render(self, mode="human"):
        print(f"t={self.idx}, price={self.prices[self.idx]}")
