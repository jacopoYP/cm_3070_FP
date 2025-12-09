import numpy as np
import gym
from gym import spaces


class SellEnv(gym.Env):
    """
    Environment for SELL decisions (closing an existing long position).

    Assumptions:
    - At reset(), we start with a LONG position already open at some entry_idx.
    - The agent can either:
        0 = HOLD (keep the position open)
        1 = SELL (close the position now)

    Episode ends when:
        - the agent chooses SELL (and closes the trade), or
        - horizon steps since entry are reached, or
        - we reach the end of the dataset.

    Option B fixes:
    - entry_idx is sampled with a buffer from the end (so episodes aren't 1 step long)
    - minimum hold time before SELL is allowed
    - horizon is respected properly
    - numerically stable reward (same style as BuyEnv)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        state_window_df,
        price_series,
        horizon: int = 20,
        transaction_cost: float = 0.001,
        lambda_dd: float = 0.05,
        lambda_vol: float = 0.01,
        hold_penalty_long: float = 0.0,
        max_episode_steps: int | None = None,
        min_steps_before_sell: int = 3,
        min_buffer_from_end: int = 30,
    ):
        super().__init__()

        self.state_df = state_window_df
        self.price_series = price_series
        self.horizon = horizon
        self.transaction_cost = transaction_cost
        self.lambda_dd = lambda_dd
        self.lambda_vol = lambda_vol
        self.hold_penalty_long = hold_penalty_long
        self.max_episode_steps = max_episode_steps

        self.min_steps_before_sell = min_steps_before_sell
        self.min_buffer_from_end = min_buffer_from_end

        self.num_states = state_window_df.shape[0]
        self.num_features = state_window_df.shape[1]

        # Action space: HOLD / SELL
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32,
        )

        # Episode state
        self.idx: int = 0
        self.entry_idx: int | None = None
        self.entry_price: float | None = None
        self.steps_since_entry: int = 0
        self.episode_steps: int = 0

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def reset(self):
        """
        For training:
        - pick a random entry_idx such that there is enough room for horizon steps
          AND a safety buffer from the end of the dataset.
        - assume we are already long at that price
        """
        # Make sure we have room: horizon + buffer
        max_start = self.num_states - self.horizon - self.min_buffer_from_end
        max_start = max(0, max_start)

        # If dataset is short, fallback to simple range
        if max_start <= 0:
            self.entry_idx = 0
        else:
            self.entry_idx = np.random.randint(0, max_start + 1)

        self.idx = self.entry_idx
        self.entry_price = float(self.price_series.iloc[self.entry_idx])
        self.steps_since_entry = 0
        self.episode_steps = 0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return self.state_df.iloc[self.idx].values.astype(np.float32)

    def step(self, action: int):
        """
        Step the environment by one time step.

        - If SELL and min_steps_before_sell satisfied: close trade now.
        - If SELL too early: treated as HOLD.
        - If HOLD:
            * advance time
            * forced close at horizon/end/max_episode_steps.
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        if self.entry_idx is None or self.entry_price is None:
            raise RuntimeError("SellEnv used without a valid open position.")

        done = False
        info = {}
        reward = 0.0

        # Enforce minimum holding time before SELL is allowed
        if action == 1 and self.steps_since_entry < self.min_steps_before_sell:
            action = 0  # override to HOLD

        current_price = float(self.price_series.iloc[self.idx])

        # ------------------------
        # Action handling
        # ------------------------
        explicitly_sold = False

        if action == 1:
            # SELL: close now
            trade_reward, trade_info = self._compute_trade_reward(
                entry_idx=self.entry_idx,
                exit_idx=self.idx,
                forced=False,
            )
            reward += trade_reward
            done = True
            explicitly_sold = True
            info.update(trade_info)
        else:
            # HOLD: small penalty while in position (optional)
            reward -= self.hold_penalty_long

        # ------------------------
        # Advance time if not done
        # ------------------------
        self.episode_steps += 1
        self.steps_since_entry += 1

        if not done:
            self.idx += 1

            # End conditions: horizon reached or end of data or max steps
            if self.idx >= self.num_states - 1:
                done = True
            if self.steps_since_entry >= self.horizon:
                done = True
            if self.max_episode_steps is not None and self.episode_steps >= self.max_episode_steps:
                done = True

        # ------------------------
        # Forced close at end/horizon (only if not already sold)
        # ------------------------
        if done and not explicitly_sold:
            exit_idx = min(self.idx, self.num_states - 1)
            forced_reward, forced_info = self._compute_trade_reward(
                entry_idx=self.entry_idx,
                exit_idx=exit_idx,
                forced=True,
            )
            reward += forced_reward
            info.update(forced_info)

        # ------------------------
        # Next state
        # ------------------------
        if not done:
            next_state = self._get_state()
        else:
            next_state = np.zeros(self.num_features, dtype=np.float32)

        return next_state, float(reward), done, info

    # ------------------------------------------------------------------ #
    # Reward computation (NaN-safe, similar to BuyEnv)
    # ------------------------------------------------------------------ #

    def _compute_trade_reward(
        self,
        entry_idx: int,
        exit_idx: int,
        forced: bool = False,
    ) -> tuple[float, dict]:
        entry_idx = max(0, min(entry_idx, self.num_states - 1))
        exit_idx = max(0, min(exit_idx, self.num_states - 1))

        entry_price = float(self.price_series.iloc[entry_idx])
        exit_price = float(self.price_series.iloc[exit_idx])

        # Gross return
        if entry_price == 0 or np.isnan(entry_price) or np.isnan(exit_price):
            gross_return = 0.0
        else:
            gross_return = (exit_price - entry_price) / entry_price

        price_window = self.price_series.iloc[entry_idx : exit_idx + 1].astype(float)

        if len(price_window) > 1:
            peak = float(price_window.max())
            trough = float(price_window.min())
            if peak == 0 or np.isnan(peak) or np.isnan(trough):
                max_drawdown = 0.0
            else:
                max_drawdown = (trough - peak) / peak

            returns = price_window.pct_change().dropna()
            if len(returns) > 1:
                volatility = float(returns.std())
            else:
                volatility = 0.0
        else:
            max_drawdown = 0.0
            volatility = 0.0

        drawdown_penalty = self.lambda_dd * abs(max_drawdown)
        volatility_penalty = self.lambda_vol * abs(volatility)

        trade_reward = (
            gross_return
            - self.transaction_cost
            - drawdown_penalty
            - volatility_penalty
        )

        info = {
            "trade_reward": trade_reward,
            "gross_return": gross_return,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "forced_close": forced,
        }

        return float(trade_reward), info

    def render(self, mode="human"):
        pass
