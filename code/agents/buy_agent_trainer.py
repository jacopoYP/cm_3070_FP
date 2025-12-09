import yaml
import numpy as np
from typing import Optional, Dict, Any, List

from agents.ddqn import DDQNAgent
from envs.buy_env import BuyEnv
from pipeline.build_dataset import make_state_frame
from features.state_assembler import StateAssembler


class BuyAgentTrainer:
    """
    Production-grade trainer for the BUY agent.

    Handles:
    - dataset creation
    - state window assembly
    - BuyEnv setup
    - DDQN training loop
    - dynamic warmup based on dataset length
    - trade-level and episode-level logging
    """

    def __init__(
        self,
        cfg_path: str = "config/data_config.yaml",
        ticker: str = "AAPL",
        window_size: int = 30,
        horizon: int = 10,
        transaction_cost: float = 0.001,
        device: Optional[str] = None,
        # lambda_dd: float = 0.5,
        # lambda_vol: float = 0.1,
        # lambda_tf: float = 0.001,
        lambda_dd: float = 0.1,     # tuned
        lambda_vol: float = 0.02,   # tuned
        lambda_tf: float = 0.0005,  # tuned
        hold_penalty_flat: float = 0.0,
        hold_penalty_long: float = 0.0,
    ):
        # Configuration
        self.cfg = yaml.safe_load(open(cfg_path))
        self.ticker = ticker
        self.window_size = window_size
        self.horizon = horizon
        self.transaction_cost = transaction_cost
        self.device = device

        # Reward parameters (GA-friendly)
        self.lambda_dd = lambda_dd
        self.lambda_vol = lambda_vol
        self.lambda_tf = lambda_tf
        self.hold_penalty_flat = hold_penalty_flat
        self.hold_penalty_long = hold_penalty_long

        # Objects to be built later
        self.env: Optional[BuyEnv] = None
        self.agent: Optional[DDQNAgent] = None
        self.state_df = None
        self.prices = None

        # Persistent training logs
        self.training_history: Dict[str, List[Any]] = {
            "episode_rewards": [],
            "epsilon": [],
            "steps": [],
            "buffer_size": [],
            "num_trades": [],
            "mean_trade_reward": [],
        }

    # ----------------------------------------------------------------------
    # Dataset + Environment Setup
    # ----------------------------------------------------------------------

    def _build_dataset_and_env(self):
        """
        1. Load dataset (prices + technical indicators)
        2. Assemble rolling window states
        3. Create BuyEnv
        4. Create DDQN agent
        """
        dataset = make_state_frame(self.ticker, self.cfg)
        print(f"[BuyTrainer] Raw dataset: {dataset.shape}")

        dataset = dataset.dropna()
        print(f"[BuyTrainer] After dropna: {dataset.shape}")

        # Split features vs target price
        feature_cols = [c for c in dataset.columns if c != "price"]

        assembler = StateAssembler(feature_cols, self.window_size)
        state_df = assembler.assemble(dataset)
        prices = dataset.loc[state_df.index, "price"]

        print(f"[BuyTrainer] Rolling state_df shape: {state_df.shape}")

        self.state_df = state_df
        self.prices = prices

        # Build Production BuyEnv
        self.env = BuyEnv(
            state_window_df=state_df,
            price_series=prices,
            horizon=self.horizon,
            transaction_cost=self.transaction_cost,
            lambda_dd=self.lambda_dd,
            lambda_vol=self.lambda_vol,
            lambda_tf=self.lambda_tf,
            hold_penalty_flat=self.hold_penalty_flat,
            hold_penalty_long=self.hold_penalty_long,
        )

        # RL Agent
        state_dim = state_df.shape[1]
        n_actions = 2  # HOLD, BUY

        self.agent = DDQNAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            gamma=0.99,
            lr=1e-3,
            batch_size=64,
            buffer_size=300_000,   # production buffer size
            target_update_freq=1_000,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=10_000,
            device=self.device,
        )

        print(f"[BuyTrainer] state_dim={state_dim}, actions={n_actions}")

    # ----------------------------------------------------------------------

    def _ensure_components(self):
        if self.env is None or self.agent is None:
            self._build_dataset_and_env()

    # ----------------------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------------------

    def train(
        self,
        n_episodes: int = 50,
        warmup_dynamic: bool = True,
        warmup_steps: int = 500,
        max_steps_per_episode: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Full training loop for the BUY agent.

        - now uses dynamic warmup (20% of dataset) by default
        - logs trade statistics from env.step() info
        """
        self._ensure_components()
        env = self.env
        agent = self.agent
        state_df_len = len(self.state_df)

        # Dynamic warmup defaults
        if warmup_dynamic:
            warmup_steps = max(200, int(0.2 * state_df_len))
            print(f"[BuyTrainer] Dynamic warmup set to: {warmup_steps}")

        # Clear logs
        for k in self.training_history.keys():
            self.training_history[k] = []

        global_step = 0

        # ------------------------------------------------------------------
        # Main Episode Loop
        # ------------------------------------------------------------------
        for ep in range(1, n_episodes + 1):
            state = env.reset()
            ep_reward = 0.0
            steps = 0
            trade_rewards = []
            done = False

            # -------------------------------------
            # Step through the episode
            # -------------------------------------
            while not done:
                action = agent.select_action(state)  # ε-greedy
                next_state, reward, done, info = env.step(action)

                # Track trade rewards if present
                if "trade_reward" in info:
                    trade_rewards.append(info["trade_reward"])
                if "forced_close_reward" in info:
                    trade_rewards.append(info["forced_close_reward"])

                # Store transition
                agent.push_transition(state, action, reward, next_state, done)

                # Update counters
                global_step += 1
                steps += 1
                ep_reward += reward

                # Training only after warmup
                if (
                    global_step > warmup_steps
                    and len(agent.replay_buffer) >= agent.batch_size
                ):
                    agent.learn_step += 1
                    agent.update()

                state = next_state

                if max_steps_per_episode and steps >= max_steps_per_episode:
                    break

            # ------------------------------------------------------------------
            # Episode End: Logging
            # ------------------------------------------------------------------
            mean_trade_reward = np.mean(trade_rewards) if len(trade_rewards) > 0 else 0.0

            self.training_history["episode_rewards"].append(ep_reward)
            self.training_history["epsilon"].append(agent.epsilon)
            self.training_history["steps"].append(steps)
            self.training_history["buffer_size"].append(len(agent.replay_buffer))
            self.training_history["num_trades"].append(env.num_trades)
            self.training_history["mean_trade_reward"].append(mean_trade_reward)

            if verbose and (ep == 1 or ep % 5 == 0):
                avg_last10 = np.mean(self.training_history["episode_rewards"][-10:])
                print(
                    f"[Episode {ep}/{n_episodes}] "
                    f"Reward={ep_reward:.4f} | "
                    f"MeanTrades={mean_trade_reward:.4f} | "
                    f"Eps={agent.epsilon:.3f} | "
                    f"Steps={steps} | "
                    f"Trades={env.num_trades} | "
                    f"Buffer={len(agent.replay_buffer)} | "
                    f"Avg10={avg_last10:.4f}"
                )

        print("\nTraining COMPLETE.")
        print("Final 5 episode rewards:", self.training_history["episode_rewards"][-5:])

        return self.training_history

    # ----------------------------------------------------------------------
    # Evaluation (Greedy Policy)
    # ----------------------------------------------------------------------

    def make_greedy_policy(self):
        def policy(env=None):
            if env is None:
                env = self.env
            assert env is not None and self.agent is not None

            state = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = self.agent.select_action(state, greedy=True)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                state = next_state

            return total_reward, steps

        return policy

    # ----------------------------------------------------------------------
    # Report Helper
    # ----------------------------------------------------------------------

    def get_training_report(self) -> Dict[str, List[Any]]:
        """Return all training logs for analysis/reporting."""
        return self.training_history
