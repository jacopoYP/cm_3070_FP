import yaml
import numpy as np

from agents.ddqn import DDQNAgent
from envs.buy_env import BuyEnv
from pipeline.build_dataset import make_state_frame
from features.state_assembler import StateAssembler

class BuyAgentTrainer:
    """
    Helper to:
    - build dataset for one ticker
    - create rolling-window states
    - wrap everything in BuyEnv
    - train a DDQN agent on BUY vs HOLD
    """

    def __init__(
        self,
        cfg_path: str = "config/data_config.yaml",
        ticker: str = "AAPL",
        window_size: int = 30,
        horizon: int = 5,
        transaction_cost: float = 0.001,
        device: str | None = None,
    ):
        self.cfg = yaml.safe_load(open(cfg_path))
        self.ticker = ticker
        self.window_size = window_size
        self.horizon = horizon
        self.transaction_cost = transaction_cost
        self.device = device

        self.env: BuyEnv | None = None
        self.agent: DDQNAgent | None = None
        self.state_df = None
        self.prices = None

    # Dataset + Env Construction

    def _build_dataset_and_env(self):
        """
        1) Build base dataset (technicals + price) using your existing pipeline
        2) Build rolling window states with StateAssembler
        3) Align prices and create BuyEnv
        """
        dataset = make_state_frame(self.ticker, self.cfg)

        print(f"BuyTrainer || Raw dataset shape for {self.ticker}: {dataset.shape}")

        # Drop any rows with NaNs just in case
        dataset = dataset.dropna()
        print(f"BuyTrainer || After dropna: {dataset.shape}")

        # Separate features and price
        feature_cols = [c for c in dataset.columns if c != "price"]

        assembler = StateAssembler(feature_cols=feature_cols, window_size=self.window_size)
        state_df = assembler.assemble(dataset)
        prices = dataset.loc[state_df.index, "price"]

        print(f"[BuyTrainer] Rolling state_df shape: {state_df.shape}")
        # Sanity checks
        assert len(state_df) == len(prices)

        self.state_df = state_df
        self.prices = prices

        # Create environment
        self.env = BuyEnv(
            state_window_df=state_df,
            price_series=prices,
            horizon=self.horizon,
            transaction_cost=self.transaction_cost,
        )

        # Create DDQN agent
        state_dim = state_df.shape[1]
        n_actions = 2  # HOLD / BUY

        self.agent = DDQNAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            gamma=0.99,
            lr=1e-3,
            batch_size=64,
            buffer_size=20_000,
            target_update_freq=500,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=5_000,
            device=self.device,
        )

        print(f"BuyTrainer || state_dim={state_dim}, n_actions={n_actions}")

    # Train method
    def train(
        self,
        n_episodes: int = 50,
        warmup_steps: int = 500,
        max_steps_per_episode: int | None = None,
    ):
        """
        Train the BUY agent for a given number of episodes.

        - n_episodes: how many full passes through the time series
        - warmup_steps: fill replay buffer before starting DDQN updates
        - max_steps_per_episode: optional hard cap per episode (usually None)
        """
        if self.env is None or self.agent is None:
            self._build_dataset_and_env()

        episode_rewards: list[float] = []

        for ep in range(1, n_episodes + 1):
            state = self.env.reset()
            ep_reward = 0.0
            steps = 0
            done = False

            while not done:
                action = self.agent.select_action(state)  # eps-greedy
                next_state, reward, done, info = self.env.step(action)

                # Store transition
                self.agent.push_transition(state, action, reward, next_state, done)

                # Increment learning step counter
                self.agent.learn_step += 1

                # Update agent after warmup
                if self.agent.learn_step > warmup_steps:
                    self.agent.update()

                state = next_state
                ep_reward += reward
                steps += 1

                if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                    break

            episode_rewards.append(ep_reward)

            if ep % 10 == 0 or ep == 1:
                print(
                    f"[Episode {ep}/{n_episodes}] "
                    f"Reward={ep_reward:.4f}, "
                    f"Epsilon={self.agent.epsilon:.3f}"
                )

        print("Training complete.")
        print("Final rewards:", episode_rewards[-10:])

        return episode_rewards

    # Greedy Policy for Evaluation
    def make_greedy_policy(self):
        """
        Returns a small function that:
        - runs one greedy episode (no exploration)
        - returns total_reward, steps
        """

        def policy(env=None) -> tuple[float, int]:
            if env is None:
                env = self.env
            assert env is not None, "Environment not built yet."
            assert self.agent is not None, "Agent not built yet."

            state = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = self.agent.select_action(state, greedy=True)
                state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1

            return total_reward, steps

        return policy