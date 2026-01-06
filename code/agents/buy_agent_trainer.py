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

from pipeline.state_builders import build_states_and_prices

from utils.entry_indices import generate_buy_entry_indices

from dataclasses import replace
from utils.dummy_agents import AlwaysHoldSellAgent
from scripts.trade_manager import TradeManager, TradeManagerParams
from scripts.metrics import max_drawdown

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

        # self.state_df = state_df
        # self.prices = prices
        state_df, prices = build_states_and_prices(self.ticker, self.config)
        self.state_df = state_df
        self.prices = prices  # now numpy float32

        # --- Env
        self.env = BuyEnv(
            features=state_df.values.astype(np.float32),
            prices=prices,
            config=self.config,
        )
        # self.env = BuyEnv(
        #     features=state_df.values,
        #     prices=prices.values,
        #     config=self.config,
        # )


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

        best_score = -1e18
        best_state = None
        best_metrics = None

        for ep in range(1, n_episodes + 1):
            state = self.env.reset()
            done = False
            ep_reward = 0.0
            steps = 0

            while not done:
                action = self.agent.select_action(state)  # eps-greedy
                next_state, reward, done, info = self.env.step(action)

                # self.agent.push_transition(state, action, reward, next_state, done)

                # # Warmup gate:
                # # NOTE: your agent.update() increments learn_step internally.
                # # We keep learn_step moving during warmup by calling update only after warmup.
                # if self.agent.learn_step > warmup_steps:
                #     self.agent.update()
                # else:
                #     self.agent.learn_step += 1
                self.agent.push_transition(state, action, reward, next_state, done)

                # always increment learn_step once per env step
                self.agent.learn_step += 1

                # only start updating after warmup AND after buffer has enough samples
                if self.agent.learn_step > warmup_steps:
                    self.agent.update()

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

            if ep % log_every == 0:
                score, metrics = self._score_buy_policy_via_trade_manager()
                print(f"[BuyTrainer] TM-checkpoint score={score:.6f} metrics={metrics}")

                if score > best_score:
                    best_score = score
                    best_metrics = metrics
                    best_state = {k: v.detach().cpu().clone() for k, v in self.agent.q_net.state_dict().items()}
                    print(f"[BuyTrainer] ✅ New BEST checkpoint at ep={ep}: {best_metrics}")


            if verbose and (ep == 1 or ep % log_every == 0):
                print(
                    f"[Buy Ep {ep}/{n_episodes}] Reward={ep_reward:.4f} | "
                    f"Eps={self.agent.epsilon:.3f} | Steps={steps} | "
                    f"Buffer={len(self.agent.replay_buffer)} | Avg10={row['avg10_reward']:.4f}"
                )

        df = pd.DataFrame(rows)

        if best_state is not None:
            self.agent.q_net.load_state_dict(best_state)
            self.agent.target_net.load_state_dict(best_state)
            print(f"[BuyTrainer] Restored BEST checkpoint: {best_metrics} (score={best_score:.6f})")
        else:
            print("[BuyTrainer] WARNING: No checkpoint captured (policy may be dead).")


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

    def collect_buy_entry_indices(self, greedy: bool = True) -> np.ndarray:
        assert self.agent is not None
        assert self.state_df is not None
        assert self.prices is not None

        tm = self.config.trade_manager
        conf_temp = float(getattr(tm, "confidence_temp", 0.01))

        return generate_buy_entry_indices(
            buy_agent=self.agent,
            state_df=self.state_df.values.astype(np.float32),
            prices=np.asarray(self.prices, dtype=np.float32),
            # buy_min_confidence=float(tm.buy_min_confidence) if getattr(tm, "buy_min_confidence", None) is not None else None,
            buy_min_confidence=None,
            confidence_temp=conf_temp,
            # use_trend_filter=bool(getattr(tm, "use_trend_filter", False)),
            use_trend_filter=False,
            ma_short=int(getattr(tm, "ma_short", 10)),
            ma_long=int(getattr(tm, "ma_long", 30)),
            # cooldown_steps=int(getattr(tm, "cooldown_steps", 0)),
            cooldown_steps=5,
        )

    def _score_buy_policy_via_trade_manager(self) -> tuple[float, dict]:
        """Returns (score, metrics). Uses AlwaysHoldSellAgent + forced time exits.
        Score penalizes drawdown and rejects 'no-trade' policies.
        """
        assert self.state_df is not None and self.prices is not None and self.agent is not None

        tm = TradeManager(
            buy_agent=self.agent,
            sell_agent=AlwaysHoldSellAgent(),
            state_df=self.state_df,     # keep as DataFrame
            prices=self.prices,         # keep as Series
            config=self.config,
            params=TradeManagerParams(
                cooldown_steps=int(self.config.trade_manager.cooldown_steps),
                transaction_cost_round_trip=float(self.config.reward.transaction_cost),
                max_hold_bars=int(self.config.trade_manager.sell_horizon),
                force_close_at_end=True,

                # IMPORTANT: use the SAME gates as evaluation (option 2)
                buy_min_confidence=float(self.config.trade_manager.buy_min_confidence),
                use_trend_filter=bool(self.config.trade_manager.use_trend_filter),
                ma_short=int(self.config.trade_manager.ma_short),
                ma_long=int(self.config.trade_manager.ma_long),
                confidence_temp=float(getattr(self.config.trade_manager, "confidence_temp", 0.01)),
            ),
        )

        res = tm.run()

        # you already have summarize_backtest; use it if available
        # otherwise compute minimal metrics here:
        trades = res["trades"]
        n_trades = len(trades)
        equity = float(res["final_equity"])

        # simple max drawdown from equity_curve:
        curve = np.asarray(res["equity_curve"], dtype=np.float32)
        peak = np.maximum.accumulate(curve)
        dd = (curve / np.maximum(peak, 1e-12)) - 1.0
        # max_dd = float(dd.min()) if len(dd) else 0.0
        max_dd = max_drawdown(curve)

        # Reject dead policies (no trades) so HOLD-only cannot "win"
        # min_trades = 5
        min_trades = 10
        if n_trades < min_trades:
            return -1e9, {"final_equity": equity, "n_trades": n_trades, "max_drawdown": max_dd}
        
        # Score: equity with drawdown penalty
        score = equity + 0.5 * max_dd   # max_dd is negative, so this penalizes DD
        return score, {"final_equity": equity, "n_trades": n_trades, "max_drawdown": max_dd}
