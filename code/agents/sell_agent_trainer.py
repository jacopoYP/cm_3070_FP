import yaml
import numpy as np
from typing import Optional, Dict, Any, List

from agents.ddqn import DDQNAgent
from envs.sell_env import SellEnv
from pipeline.build_dataset import make_state_frame
from features.state_assembler import StateAssembler


class SellAgentTrainer:
    """
    Trainer for SELL agent (when to close an existing long position).

    Uses SellEnv and focuses on exit decisions only.
    """

    def __init__(
        self,
        cfg_path: str = "config/data_config.yaml",
        ticker: str = "AAPL",
        window_size: int = 30,
        horizon: int = 20,
        transaction_cost: float = 0.001,
        device: Optional[str] = None,
        lambda_dd: float = 0.05,
        lambda_vol: float = 0.01,
        hold_penalty_long: float = 0.0,
    ):
        self.cfg = yaml.safe_load(open(cfg_path))
        self.ticker = ticker
        self.window_size = window_size
        self.horizon = horizon
        self.transaction_cost = transaction_cost
        self.device = device

        self.lambda_dd = lambda_dd
        self.lambda_vol = lambda_vol
        self.hold_penalty_long = hold_penalty_long

        self.env: Optional[SellEnv] = None
        self.agent: Optional[DDQNAgent] = None
        self.state_df = None
        self.prices = None

        self.training_history: Dict[str, List[Any]] = {
            "episode_rewards": [],
            "epsilon": [],
            "steps": [],
            "buffer_size": [],
            "mean_trade_reward": [],
        }

    # ------------------------------------------------------------------ #
    # Dataset + Env
    # ------------------------------------------------------------------ #

    def _build_dataset_and_env(self):
        dataset = make_state_frame(self.ticker, self.cfg)
        print(f"[SellTrainer] Raw dataset: {dataset.shape}")

        dataset = dataset.dropna()
        print(f"[SellTrainer] After dropna: {dataset.shape}")

        feature_cols = [c for c in dataset.columns if c != "price"]
        assembler = StateAssembler(feature_cols, self.window_size)

        state_df = assembler.assemble(dataset)
        prices = dataset.loc[state_df.index, "price"]

        print(f"[SellTrainer] state_df shape: {state_df.shape}")

        self.state_df = state_df
        self.prices = prices

        self.env = SellEnv(
            state_window_df=state_df,
            price_series=prices,
            horizon=self.horizon,
            transaction_cost=self.transaction_cost,
            lambda_dd=self.lambda_dd,
            lambda_vol=self.lambda_vol,
            hold_penalty_long=self.hold_penalty_long,
            # we keep defaults for min_steps_before_sell and min_buffer_from_end
        )

        state_dim = state_df.shape[1]
        n_actions = 2  # HOLD / SELL

        self.agent = DDQNAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            gamma=0.99,
            lr=1e-3,
            batch_size=64,
            buffer_size=300_000,
            target_update_freq=1_000,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=2_000,  # much faster decay than 10_000
            device=self.device,
        )

        print(f"[SellTrainer] state_dim={state_dim}, actions={n_actions}")

    def _ensure_components(self):
        if self.env is None or self.agent is None:
            self._build_dataset_and_env()

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def train(
        self,
        n_episodes: int = 50,
        warmup_dynamic: bool = True,
        warmup_steps: int = 500,
        max_steps_per_episode: Optional[int] = None,
        verbose: bool = True,
    ):
        self._ensure_components()
        env = self.env
        agent = self.agent
        state_len = len(self.state_df)

        if warmup_dynamic:
            warmup_steps = max(200, int(0.2 * state_len))
            print(f"[SellTrainer] Dynamic warmup set to: {warmup_steps}")

        for k in self.training_history.keys():
            self.training_history[k] = []

        global_step = 0

        for ep in range(1, n_episodes + 1):
            state = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0
            trade_rewards = []

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                if "trade_reward" in info:
                    trade_rewards.append(info["trade_reward"])

                agent.push_transition(state, action, reward, next_state, done)

                global_step += 1
                steps += 1
                ep_reward += reward

                if (
                    global_step > warmup_steps
                    and len(agent.replay_buffer) >= agent.batch_size
                ):
                    agent.learn_step += 1
                    agent.update()

                state = next_state

                if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                    break

            mean_trade_reward = np.mean(trade_rewards) if trade_rewards else 0.0

            self.training_history["episode_rewards"].append(ep_reward)
            self.training_history["epsilon"].append(agent.epsilon)
            self.training_history["steps"].append(steps)
            self.training_history["buffer_size"].append(len(agent.replay_buffer))
            self.training_history["mean_trade_reward"].append(mean_trade_reward)

            if verbose and (ep == 1 or ep % 5 == 0):
                avg_last10 = np.mean(self.training_history["episode_rewards"][-10:])
                print(
                    f"[Sell Ep {ep}/{n_episodes}] "
                    f"Reward={ep_reward:.4f} | "
                    f"MeanTrade={mean_trade_reward:.4f} | "
                    f"Eps={agent.epsilon:.3f} | "
                    f"Steps={steps} | "
                    f"Buffer={len(agent.replay_buffer)} | "
                    f"Avg10={avg_last10:.4f}"
                )

        print("\nSellAgent Training COMPLETE.")
        print("Final 5 episode rewards:", self.training_history["episode_rewards"][-5:])

        return self.training_history

    # ------------------------------------------------------------------ #
    # Greedy policy for evaluation
    # ------------------------------------------------------------------ #

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

    def get_training_report(self) -> Dict[str, List[Any]]:
        return self.training_history
