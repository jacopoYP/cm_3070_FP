import numpy as np
import gym
from gym import spaces


class BuyEnv(gym.Env):
    """
    Production BUY environment.

    Actions:
        0 = HOLD
        1 = BUY

    Behaviour:
    - When flat and agent selects BUY (1):
        -> open a long position at current price.
    - When already long and agent selects BUY:
        -> treated as HOLD (no scaling/leverage for now).

    - Position is automatically closed when:
        -> holding_steps >= horizon, or
        -> we reach the end of the dataset.

    Reward:
    - 0 (or tiny penalties) on normal HOLD steps.
    - On CLOSE (normal or forced at episode end):
        reward = gross_return - transaction_cost
                 - lambda_dd * |max_drawdown|
                 - lambda_vol * volatility
                 - lambda_tf * num_trades

    Extra:
    - Cooldown period after each close, during which agent is forced to HOLD.
    - NaN-safe for all metrics.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        state_window_df,
        price_series,
        horizon: int = 10,
        transaction_cost: float = 0.001,
        # Tuned penalties (milder than before)
        lambda_dd: float = 0.1,      # drawdown penalty scale
        lambda_vol: float = 0.02,    # volatility penalty scale
        lambda_tf: float = 0.0005,   # trade frequency penalty scale
        # Per-step penalties (optional, can be 0)
        hold_penalty_flat: float = 0.0,
        hold_penalty_long: float = 0.0,
        # Episode control
        max_episode_steps: int | None = None,
        # Cooldown after closing a trade
        cooldown_period: int = 5,
    ):
        super().__init__()

        self.state_df = state_window_df
        self.price_series = price_series
        self.horizon = horizon
        self.transaction_cost = transaction_cost
        self.lambda_dd = lambda_dd
        self.lambda_vol = lambda_vol
        self.lambda_tf = lambda_tf
        self.hold_penalty_flat = hold_penalty_flat
        self.hold_penalty_long = hold_penalty_long

        self.num_states = state_window_df.shape[0]
        self.num_features = state_window_df.shape[1]

        # Gym spaces
        self.action_space = spaces.Discrete(2)  # 0 = HOLD, 1 = BUY
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32,
        )

        # Episode internals
        self.idx: int = 0
        self.position: int = 0  # 0 = flat, 1 = long
        self.entry_price: float | None = None
        self.entry_idx: int | None = None
        self.num_trades: int = 0
        self.episode_steps: int = 0
        self.max_episode_steps = max_episode_steps

        # Cooldown state
        self.cooldown_period = cooldown_period
        self.cooldown_steps: int = 0

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def reset(self):
        self.idx = 0
        self.position = 0
        self.entry_price = None
        self.entry_idx = None
        self.num_trades = 0
        self.episode_steps = 0
        self.cooldown_steps = 0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return self.state_df.iloc[self.idx].values.astype(np.float32)

    def step(self, action: int):
        """
        Step the environment by one time step.

        1) Apply cooldown override (forces HOLD).
        2) Apply action given current position.
        3) Check close conditions, compute trade reward if closing.
        4) Advance time, maybe force close at the end.
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        done = False
        info = {}
        reward = 0.0

        # ------------------------------------------------------------------
        # 0) Cooldown override: force HOLD if cooling down
        # ------------------------------------------------------------------
        if self.cooldown_steps > 0:
            action = 0  # HOLD
            self.cooldown_steps -= 1

        current_price = self.price_series.iloc[self.idx]

        # ------------------------------------------------------------------
        # 1) Apply Action
        # ------------------------------------------------------------------
        if self.position == 0:
            # FLAT
            if action == 1:
                # Open new long
                self.position = 1
                self.entry_price = float(current_price)
                self.entry_idx = self.idx
                self.num_trades += 1
                # We apply transaction cost at close, not here.
            else:
                # HOLD while flat
                reward -= self.hold_penalty_flat
        else:
            # LONG (BUY treated as HOLD for now)
            reward -= self.hold_penalty_long

        # ------------------------------------------------------------------
        # 2) Check Close Conditions (normal close)
        # ------------------------------------------------------------------
        close_position = False

        if self.position == 1 and self.entry_idx is not None:
            holding_steps = self.idx - self.entry_idx
            if holding_steps >= self.horizon or self.idx >= self.num_states - 1:
                close_position = True

        if close_position and self.position == 1 and self.entry_idx is not None:
            trade_reward, trade_info = self._compute_trade_reward(
                entry_idx=self.entry_idx,
                exit_idx=self.idx,
            )
            reward += trade_reward

            # Reset position
            self.position = 0
            self.entry_price = None
            self.entry_idx = None

            # Start cooldown after close
            self.cooldown_steps = self.cooldown_period

            # Merge trade info into info dict
            info.update(trade_info)

        # ------------------------------------------------------------------
        # 3) Advance Time
        # ------------------------------------------------------------------
        self.episode_steps += 1
        self.idx += 1

        # Episode termination conditions
        if self.idx >= self.num_states - 1:
            done = True
        if self.max_episode_steps is not None and self.episode_steps >= self.max_episode_steps:
            done = True

        # ------------------------------------------------------------------
        # 4) Force close if episode ended while still LONG
        # ------------------------------------------------------------------
        if done and self.position == 1 and self.entry_idx is not None:
            # Force close at final valid index
            final_idx = min(self.idx, self.num_states - 1)
            closing_reward, closing_info = self._compute_trade_reward(
                entry_idx=self.entry_idx,
                exit_idx=final_idx,
                forced=True,
            )
            reward += closing_reward

            # Clear position
            self.position = 0
            self.entry_price = None
            self.entry_idx = None

            info.update(closing_info)

        # ------------------------------------------------------------------
        # 5) Next State
        # ------------------------------------------------------------------
        if not done:
            next_state = self._get_state()
        else:
            next_state = np.zeros(self.num_features, dtype=np.float32)

        return next_state, float(reward), done, info

    # ------------------------------------------------------------------ #
    # Reward computation (NaN-safe)
    # ------------------------------------------------------------------ #

    def _compute_trade_reward(
        self,
        entry_idx: int,
        exit_idx: int,
        forced: bool = False,
    ) -> tuple[float, dict]:
        """
        Compute risk-adjusted reward for a trade between entry_idx and exit_idx.
        Fully NaN-safe.
        """
        # Clamp indices
        entry_idx = max(0, min(entry_idx, self.num_states - 1))
        exit_idx = max(0, min(exit_idx, self.num_states - 1))

        entry_price = float(self.price_series.iloc[entry_idx])
        exit_price = float(self.price_series.iloc[exit_idx])

        # Gross return
        if entry_price == 0 or np.isnan(entry_price) or np.isnan(exit_price):
            gross_return = 0.0
        else:
            gross_return = (exit_price - entry_price) / entry_price

        # Price window for risk metrics
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

        # Penalties
        drawdown_penalty = self.lambda_dd * abs(max_drawdown)
        volatility_penalty = self.lambda_vol * abs(volatility)
        frequency_penalty = self.lambda_tf * self.num_trades

        trade_reward = (
            gross_return
            - self.transaction_cost
            - drawdown_penalty
            - volatility_penalty
            - frequency_penalty
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
