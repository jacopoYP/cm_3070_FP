from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.system import TradingSystemConfig
from agents.ddqn import DDQNAgent
from envs.buy_env import BuyEnv

from pipeline.build_dataset import make_state_frame
from features.state_assembler import StateAssembler
# import yaml
# import numpy as np
# import torch

# from agents.ddqn import DDQNAgent
# from envs.buy_env import BuyEnv
# from pipeline.build_dataset import make_state_frame
# from features.state_assembler import StateAssembler

# from agents.multi_process.multi_process_trainer import MultiProcessTrainer
# from agents.multi_process.handler import EnvHandler, AgentHandler


# class BuyAgentTrainer:
#     """
#     Helper to:
#     - build dataset for one ticker
#     - create rolling-window states
#     - wrap everything in BuyEnv
#     - train a DDQN Buy Agent (standard or trend-filtered)
#     """

#     def __init__(
#         self,
#         cfg_path: str = "config/data_config.yaml",
#         ticker: str = "AAPL",
#         window_size: int = 30,
#         horizon: int = 20,
#         transaction_cost: float = 0.001,
#         lambda_dd: float = 0.05,
#         lambda_vol: float = 0.01,
#         hold_penalty_long: float = 0.0,
#         device: str | None = None,
#     ):
#         self.cfg = yaml.safe_load(open(cfg_path))
#         self.ticker = ticker
#         self.window_size = window_size
#         self.horizon = horizon
#         self.transaction_cost = transaction_cost

#         self.lambda_dd = lambda_dd
#         self.lambda_vol = lambda_vol
#         self.hold_penalty_long = hold_penalty_long

#         self.device = device

#         self.env: BuyEnv | None = None
#         self.agent: DDQNAgent | None = None
#         self.state_df = None
#         self.prices = None

#         # will hold boolean trend mask aligned with state_df
#         self.trend_mask: np.ndarray | None = None

#     # ------------------------------------------------------------------
#     # Dataset + Env
#     # ------------------------------------------------------------------
#     def _build_dataset_and_env(self):
#         dataset = make_state_frame(self.ticker, self.cfg)
#         print(f"[BuyTrainer] Raw dataset: {dataset.shape}")
#         dataset = dataset.dropna()
#         print(f"[BuyTrainer] After dropna: {dataset.shape}")

#         feature_cols = [c for c in dataset.columns if c != "price"]

#         assembler = StateAssembler(feature_cols=feature_cols, window_size=self.window_size)
#         state_df = assembler.assemble(dataset)
#         prices = dataset.loc[state_df.index, "price"]

#         print(f"[BuyTrainer] Rolling state_df shape: {state_df.shape}")

#         self.state_df = state_df
#         self.prices = prices

#         # Build BuyEnv
#         self.env = BuyEnv(
#             state_window_df=state_df,
#             price_series=prices,
#             horizon=self.horizon,
#             transaction_cost=self.transaction_cost,
#             lambda_dd=self.lambda_dd,
#             lambda_vol=self.lambda_vol,
#             hold_penalty_long=self.hold_penalty_long,
#         )

#         # Build agent
#         state_dim = state_df.shape[1]
#         n_actions = 2  # HOLD / BUY
#         self.agent = DDQNAgent(
#             state_dim=state_dim,
#             n_actions=n_actions,
#             gamma=0.99,
#             lr=1e-3,
#             batch_size=64,
#             buffer_size=200_000,
#             target_update_freq=500,
#             epsilon_start=1.0,
#             epsilon_end=0.05,
#             epsilon_decay_steps=10_000,
#             device=self.device,
#         )

#         print(f"[BuyTrainer] state_dim={state_dim}, actions={n_actions}")

#         # compute trend mask once
#         self._compute_trend_mask()

#     # ------------------------------------------------------------------
#     # Trend filter (SMA-based, like TradeManager)
#     # ------------------------------------------------------------------
#     def _rolling_mean(self, arr: np.ndarray, window: int) -> np.ndarray:
#         if window <= 1:
#             return arr.copy()
#         cumsum = np.cumsum(np.insert(arr, 0, 0.0))
#         out = (cumsum[window:] - cumsum[:-window]) / float(window)
#         pad = np.full(window - 1, np.nan, dtype=np.float32)
#         return np.concatenate([pad, out])

#     def _compute_trend_mask(self):
#         """
#         Build a boolean mask where trend is bullish (SMA short > SMA long).
#         This is aligned with state_df index and stored as self.trend_mask.
#         """
#         if self.state_df is None or self.prices is None:
#             return

#         tm_cfg = self.cfg.get("trade_manager", {})
#         ma_short = tm_cfg.get("ma_short", 10)
#         ma_long = tm_cfg.get("ma_long", 30)

#         prices_np = self.prices.values.astype(np.float32)
#         sma_s = self._rolling_mean(prices_np, ma_short)
#         sma_l = self._rolling_mean(prices_np, ma_long)

#         mask = ~np.isnan(sma_s) & ~np.isnan(sma_l) & (sma_s > sma_l)
#         # align to state_df index, which is subset of full dataset
#         # here we assume state_df and prices share the same order
#         self.trend_mask = mask.astype(bool)
#         print(
#             f"[BuyTrainer] Trend mask computed. True count = {self.trend_mask.sum()} "
#             f"out of {len(self.trend_mask)}"
#         )

#     # ------------------------------------------------------------------
#     # Standard single-process training (unfiltered)
#     # ------------------------------------------------------------------
#     def train(
#         self,
#         n_episodes: int = 50,
#         warmup_dynamic: bool = True,
#         warmup_steps: int = 500,
#         verbose: bool = True,
#     ):
#         if self.env is None or self.agent is None:
#             self._build_dataset_and_env()

#         if warmup_dynamic:
#             warmup_steps = max(200, int(0.2 * len(self.state_df)))
#             if verbose:
#                 print(f"[BuyTrainer] Dynamic warmup = {warmup_steps}")

#         episode_rewards: list[float] = []

#         for ep in range(1, n_episodes + 1):
#             state = self.env.reset()
#             done = False
#             ep_reward = 0.0
#             steps = 0

#             while not done:
#                 action = self.agent.select_action(state)
#                 next_state, reward, done, info = self.env.step(action)

#                 self.agent.push_transition(state, action, reward, next_state, done)
#                 self.agent.learn_step += 1

#                 if self.agent.learn_step > warmup_steps:
#                     self.agent.update()

#                 state = next_state
#                 ep_reward += reward
#                 steps += 1

#             episode_rewards.append(ep_reward)

#             if verbose and (ep == 1 or ep % 10 == 0):
#                 print(
#                     f"[Buy Ep {ep}/{n_episodes}] "
#                     f"Reward={ep_reward:.4f} | Steps={steps} | "
#                     f"Eps={self.agent.epsilon:.3f} | Buffer={len(self.agent.replay_buffer)}"
#                 )

#         print("BuyAgent Training COMPLETE.")
#         print("Final 5 episode rewards:", episode_rewards[-5:])
#         return episode_rewards

#     # ------------------------------------------------------------------
#     # NEW: Trend-filtered single-process training
#     # ------------------------------------------------------------------
#     def train_trend_filtered(
#         self,
#         n_episodes: int = 200,
#         warmup_dynamic: bool = True,
#         warmup_steps: int = 500,
#         verbose: bool = True,
#     ):
#         """
#         Train the BuyAgent but only store transitions that are:
#         - in a bullish trend region (trend_mask True), OR
#         - have non-zero reward (trade exit)

#         This increases the signal-to-noise ratio without changing BuyEnv.
#         """
#         if self.env is None or self.agent is None:
#             self._build_dataset_and_env()

#         if self.trend_mask is None:
#             self._compute_trend_mask()

#         assert self.trend_mask is not None

#         if warmup_dynamic:
#             warmup_steps = max(200, int(0.2 * len(self.state_df)))
#             if verbose:
#                 print(f"[BuyTrainer] Trend-filtered warmup = {warmup_steps}")

#         episode_rewards: list[float] = []

#         for ep in range(1, n_episodes + 1):
#             state = self.env.reset()
#             done = False
#             ep_reward = 0.0
#             steps = 0
#             t = 0  # position in state_df

#             while not done:
#                 # standard eps-greedy
#                 action = self.agent.select_action(state)
#                 next_state, reward, done, info = self.env.step(action)

#                 # Decide whether this transition is "interesting" enough to store
#                 in_bullish_region = (
#                     0 <= t < len(self.trend_mask) and self.trend_mask[t]
#                 )
#                 has_signal = (reward != 0.0)

#                 if in_bullish_region or has_signal:
#                     self.agent.push_transition(state, action, reward, next_state, done)

#                     if self.agent.learn_step > warmup_steps:
#                         self.agent.update()
#                     self.agent.learn_step += 1

#                 state = next_state
#                 ep_reward += reward
#                 steps += 1
#                 t += 1

#             episode_rewards.append(ep_reward)

#             if verbose and (ep == 1 or ep % 10 == 0):
#                 print(
#                     f"[Buy (trend) Ep {ep}/{n_episodes}] "
#                     f"Reward={ep_reward:.4f} | Steps={steps} | "
#                     f"Eps={self.agent.epsilon:.3f} | "
#                     f"Buffer={len(self.agent.replay_buffer)}"
#                 )

#         print("BuyAgent Trend-Filtered Training COMPLETE.")
#         print("Final 5 episode rewards:", episode_rewards[-5:])
#         return episode_rewards

#     # ------------------------------------------------------------------
#     # Multi-process training (unchanged, uses EnvFactory)
#     # ------------------------------------------------------------------
#     def train_multiprocess(
#         self,
#         n_batches: int = 300,
#         updates_per_batch: int = 40,
#         n_workers: int = 4,
#         steps_per_batch: int = 300,
#     ):
#         """
#         Standard MP training (unfiltered). You can still use this if you want,
#         but for now focus on train_trend_filtered() to fix the signal.
#         """
#         if self.env is None or self.agent is None:
#             self._build_dataset_and_env()

#         EnvClass = self.env.__class__
#         env_fn = EnvHandler(
#             state_df=self.state_df,
#             prices=self.prices,
#             EnvClass=EnvClass,
#             horizon=self.horizon,
#             transaction_cost=self.transaction_cost,
#         )
#         agent_fn = AgentHandler(self.agent)

#         mp_trainer = MultiProcessTrainer(
#             agent=self.agent,
#             env_fn=env_fn,
#             agent_fn=agent_fn,
#             n_workers=n_workers,
#             steps_per_batch=steps_per_batch,
#         )

#         print("[BuyTrainer-MP] Starting MP training...")
#         mp_trainer.train(n_batches=n_batches, updates_per_batch=updates_per_batch)
#         print(
#             f"[BuyTrainer-MP] Done. Buffer size={len(self.agent.replay_buffer)} | "
#             f"Loss entries={len(self.agent.loss_history)}"
#         )




class BuyAgentTrainer:
    """
    Build dataset + rolling states, wrap into BuyEnv, train a DDQN BUY-vs-HOLD agent.

    Single-process:
        train(...)

    Multiprocess (optional, faster):
        train_multiprocess(...)
    """

    def __init__(
        self,
        ticker: str,
        config: TradingSystemConfig,
        # window_size: int = 30,
        # horizon: Optional[int] = None,
        device: str = "cpu",
    ):
        self.ticker = ticker
        self.config = config
        # self.window_size = window_size
        # self.horizon = horizon if horizon is not None else config.trade_manager.sell_horizon
        self.device = device

        self.env: Optional[BuyEnv] = None
        self.agent: Optional[DDQNAgent] = None
        self.state_df: Optional[pd.DataFrame] = None
        self.prices: Optional[pd.Series] = None

        self._build_dataset_env_agent()

    # ---------------------------------------------------------------------
    # Build dataset + env + agent
    # ---------------------------------------------------------------------

    def _build_dataset_env_agent(self) -> None:
        """
        1) dataset = make_state_frame(ticker, config)
        2) rolling window -> state_df
        3) env = BuyEnv(state_df, prices, config, horizon)
        4) agent = DDQNAgent(agent_cfg, device)
        """
        dataset = make_state_frame(self.ticker, self.config)
        print(f"[BuyTrainer] Raw dataset: {dataset.shape}")

        dataset = dataset.dropna()
        print(f"[BuyTrainer] After dropna: {dataset.shape}")

        self.feature_cols = [c for c in dataset.columns if c != "price"]

        # assembler = StateAssembler(feature_cols=feature_cols, window_size=self.window_size)
        assembler = StateAssembler(feature_cols=self.feature_cols, window_size=self.config.state.window_size)
        state_df = assembler.assemble(dataset)
        prices = dataset.loc[state_df.index, "price"]

        print(f"[BuyTrainer] Rolling state_df shape: {state_df.shape}")
        assert len(state_df) == len(prices), "State/price alignment mismatch."

        self.state_df = state_df
        self.prices = prices

        # --- Env
        # self.env = BuyEnv(
        #     state_window_df=state_df,
        #     price_series=prices,
        #     config=self.config,
        #     horizon=self.horizon,
        # )
        self.env = BuyEnv(
            features=state_df.values,
            prices=prices.values,
            config=self.config,
        )


        # --- Agent (inject computed state_dim)
        agent_cfg = replace(self.config.agent, state_dim=state_df.shape[1], n_actions=2)
        self.agent = DDQNAgent(cfg=agent_cfg, device=self.device)

        print(f"[BuyTrainer] state_dim={agent_cfg.state_dim}, actions={agent_cfg.n_actions}")

    # Exposing feature_cols
    def get_feature_cols(self):
        return self.feature_cols

    # ---------------------------------------------------------------------
    # Warmup helper
    # ---------------------------------------------------------------------

    def _dynamic_warmup(self) -> int:
        """
        Recommended: 20% of dataset, min 200.
        """
        assert self.state_df is not None
        n = len(self.state_df)
        return max(200, int(0.2 * n))

    # ---------------------------------------------------------------------
    # Training (single-process)
    # ---------------------------------------------------------------------

    def train(
        self,
        n_episodes: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        warmup_dynamic: bool = True,
        max_steps_per_episode: Optional[int] = None,
        verbose: bool = True,
        log_every: int = 10,
    ) -> Dict[str, Any]:
        """
        Single-process training.

        Returns:
          history: dict with:
            - df: pandas DataFrame (episode metrics)
            - episode_rewards: list
            - loss_history: list
        """
        assert self.env is not None and self.agent is not None

        if n_episodes is None:
            n_episodes = self.config.training.episodes

        if warmup_steps is None:
            warmup_steps = self._dynamic_warmup() if warmup_dynamic else self.config.training.warmup_steps

        print(f"[BuyTrainer] Warmup set to: {warmup_steps}")

        rows: List[Dict[str, Any]] = []
        episode_rewards: List[float] = []

        for ep in range(1, n_episodes + 1):
            state = self.env.reset()
            done = False
            ep_reward = 0.0
            steps = 0

            while not done:
                action = self.agent.select_action(state)  # eps-greedy
                next_state, reward, done, info = self.env.step(action)

                self.agent.push_transition(state, action, reward, next_state, done)

                # Warmup gate:
                # NOTE: your agent.update() increments learn_step internally.
                # We keep learn_step moving during warmup by calling update only after warmup.
                if self.agent.learn_step > warmup_steps:
                    self.agent.update()
                else:
                    self.agent.learn_step += 1

                state = next_state
                ep_reward += float(reward)
                steps += 1

                if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                    break

            episode_rewards.append(ep_reward)

            row = {
                "episode": ep,
                "episode_reward": ep_reward,
                "epsilon": float(self.agent.epsilon),
                "steps": steps,
                "buffer_size": len(self.agent.replay_buffer),
                "avg10_reward": float(np.mean(episode_rewards[-10:])),
            }
            rows.append(row)

            if verbose and (ep == 1 or ep % log_every == 0):
                print(
                    f"[Buy Ep {ep}/{n_episodes}] Reward={ep_reward:.4f} | "
                    f"Eps={self.agent.epsilon:.3f} | Steps={steps} | "
                    f"Buffer={len(self.agent.replay_buffer)} | Avg10={row['avg10_reward']:.4f}"
                )

        df = pd.DataFrame(rows)
        return {
            "df": df,
            "episode_rewards": episode_rewards,
            "loss_history": list(self.agent.loss_history),
        }

    # ---------------------------------------------------------------------
    # Greedy evaluation
    # ---------------------------------------------------------------------

    def evaluate(self, greedy: bool = True) -> Tuple[float, int]:
        """
        Runs a single episode with greedy=True (no exploration).
        Returns (total_reward, steps).
        """
        assert self.env is not None and self.agent is not None

        state = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = self.agent.select_action(state, greedy=greedy)
            state, reward, done, info = self.env.step(action)
            total_reward += float(reward)
            steps += 1

        return total_reward, steps

    # ---------------------------------------------------------------------
    # Multiprocess training (optional)
    # ---------------------------------------------------------------------

    def train_multiprocess(
        self,
        n_batches: int = 300,
        updates_per_batch: int = 40,
        n_workers: int = 4,
        steps_per_batch: int = 300,
        log_every: int = 10,
    ) -> None:
        """
        Multiprocess training = parallel experience collection, single learner (this process).

        Requirements (pickling safe on macOS spawn):
          - env_fn and agent_fn must be picklable TOP-LEVEL objects
          - do NOT pass lambdas or nested functions
        """
        assert self.env is not None and self.agent is not None
        assert self.state_df is not None and self.prices is not None

        # Local imports reduce spawn issues
        from agents.multi_process.multi_process_trainer import MultiProcessTrainer
        # from agents.multi_process.env_factory import EnvFactory
        from agents.multi_process.handler import EnvHandler, AgentHandler

        env_fn = EnvHandler(
            env_type="buy",
            state_window_df=self.state_df,
            price_series=self.prices,
            config=self.config,
            horizon=self.config.trade_manager.sell_horizon
        )
        agent_fn = AgentHandler(self.agent)

        mp_trainer = MultiProcessTrainer(
            agent=self.agent,
            env_fn=env_fn,
            agent_fn=agent_fn,
            n_workers=n_workers,
            steps_per_batch=steps_per_batch,
            log_every=log_every,
        )

        mp_trainer.train(n_batches=n_batches, updates_per_batch=updates_per_batch)

        print(
            f"[BuyTrainer-MP] Done. Replay buffer size={len(self.agent.replay_buffer)} "
            f"| Loss entries={len(self.agent.loss_history)}"
        )

    def collect_buy_entry_indices(
        self,
        greedy: bool = True,
        buy_min_confidence: float | None = None,
        use_trend_filter: bool | None = None,
    ) -> np.ndarray:
        """
        Roll forward once through the dataset and collect entry indices where the BuyAgent opens a trade.

        - greedy=True uses select_action(..., greedy=True)
        - buy_min_confidence/use_trend_filter override config temporarily for collection
        """

        # Safety check
        if self.env is None or self.agent is None:
            self._build_dataset_and_env()

        assert self.env is not None
        assert self.agent is not None

        # Use config defaults unless overridden
        tm_cfg = self.config.trade_manager
        min_conf = tm_cfg.buy_min_confidence if buy_min_confidence is None else buy_min_confidence
        trend_on = tm_cfg.use_trend_filter if use_trend_filter is None else use_trend_filter

        # If you have a trend mask somewhere, re-create it here consistently
        # (same logic used in TradeManager / training)
        prices = self.prices
        if trend_on:
            ma_s = tm_cfg.ma_short
            ma_l = tm_cfg.ma_long
            ma_short = np.convolve(prices, np.ones(ma_s)/ma_s, mode="same")
            ma_long  = np.convolve(prices, np.ones(ma_l)/ma_l, mode="same")
            trend_mask = ma_short > ma_long
        else:
            trend_mask = np.ones(len(prices), dtype=bool)

        entry_indices = []

        state = self.env.reset()
        done = False

        while not done:
            t = self.env.t  # BuyEnv uses self.t as time index
            # Get BUY “confidence” from Q-values (stable + reportable)
            # Convert Q(s,a) to pseudo-prob via softmax
            q = self.agent.q_values(state)  # <-- we’ll add this helper in Patch 2 if missing
            q = np.asarray(q, dtype=np.float32)
            expq = np.exp(q - np.max(q))
            p = expq / (np.sum(expq) + 1e-12)
            buy_conf = float(p[1])

            # Gate by trend + confidence
            if trend_mask[t] and buy_conf >= min_conf:
                action = 1
            else:
                action = 0

            next_state, reward, done, info = self.env.step(action)

            # Record entry when we actually opened a position
            # BuyEnv opens when action==1 and not already open
            if action == 1 and info.get("opened", False):
                entry_indices.append(t)

            state = next_state

        return np.array(entry_indices, dtype=int)

