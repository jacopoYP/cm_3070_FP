# # agents/buy_agent_ddqn.py
# from stable_baselines3 import DQN

# class BuyAgentDDQN:
#     """
#     Thin wrapper around Stable-Baselines3 DQN for the BuyEnv.

#     - Uses MLP policy
#     - Can be trained, saved, loaded
#     """

#     def __init__(self, env, learning_rate=1e-4, gamma=0.99, buffer_size=50_000):
#         self.env = env
#         self.model = DQN(
#             "MlpPolicy",
#             env,
#             learning_rate=learning_rate,
#             gamma=gamma,
#             buffer_size=buffer_size,
#             verbose=1,
#             tensorboard_log=None,
#         )

#     def train(self, timesteps: int = 50_000):
#         """
#         Train the agent for a given number of timesteps.
#         """
#         self.model.learn(total_timesteps=timesteps)

#     def act(self, obs):
#         """
#         Choose an action given an observation (greedy).
#         """
#         action, _ = self.model.predict(obs, deterministic=True)
#         return int(action)

#     def save(self, path: str):
#         self.model.save(path)

#     @classmethod
#     def load(cls, path: str, env):
#         obj = cls(env)
#         obj.model = DQN.load(path, env=env)
#         return obj
# agents/buy_agent_ddqn.py

import yaml
import numpy as np

from agents.ddqn import DDQNAgent
from envs.buy_env import BuyEnv
from pipeline.build_dataset import make_state_frame
# from state_assembler import StateAssembler
from pipeline.state_assembler import StateAssembler


class BuyAgentTrainer:
    """
    Small helper class to:
    - build the dataset for one ticker
    - create rolling-window states
    - wrap everything in BuyEnv
    - train a DDQN agent
    """

    def __init__(
        self,
        cfg_path: str = "config/data_config.yaml",
        ticker: str = "AAPL",
        window_size: int = 30,
        horizon: int = 5,
        transaction_cost: float = 0.001,
    ):
        self.cfg = yaml.safe_load(open(cfg_path))
        self.ticker = ticker
        self.window_size = window_size
        self.horizon = horizon
        self.transaction_cost = transaction_cost

        # 1) build base feature frame (OHLC + indicators + price)
        df = make_state_frame(self.ticker, self.cfg)
        print(f"[BuyTrainer] Raw dataset shape for {ticker}: {df.shape}")

        # choose all columns except 'price' as features
        self.feature_cols = [c for c in df.columns if c != "price"]

        # 2) build rolling-window states
        assembler = StateAssembler(self.feature_cols, window_size=self.window_size)
        state_df = assembler.assemble(df)
        print(f"[BuyTrainer] Rolling state_df shape: {state_df.shape}")

        # 3) align price vector with states
        aligned_prices = df["price"].iloc[self.window_size :]
        assert len(aligned_prices) == len(state_df)

        self.env = BuyEnv(
            state_window_df=state_df,
            price_series=aligned_prices,
            horizon=self.horizon,
            transaction_cost=self.transaction_cost,
        )

        state_dim = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        print(f"[BuyTrainer] state_dim={state_dim}, n_actions={n_actions}")

        # 4) create DDQN agent
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
        )

    def train(
        self,
        n_episodes: int = 100,
        max_steps_per_episode: int | None = None,
        warmup_steps: int = 1_000,
    ):
        """
        Simple training loop:
        - runs episodes over the BuyEnv time series
        - fills replay buffer (warmup)
        - then starts training DDQN
        """

        print("[BuyTrainer] Starting training...")
        total_steps = 0
        episode_rewards = []

        for ep in range(n_episodes):
            state = self.env.reset()
            ep_reward = 0.0
            steps = 0
            done = False

            while not done:
                # Decide action
                action = self.agent.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                # Store transition
                self.agent.push_transition(state, action, reward, next_state, done)

                state = next_state
                ep_reward += reward
                self.agent.total_steps += 1
                total_steps += 1
                steps += 1

                # Only start learning after warmup
                if self.agent.total_steps > warmup_steps:
                    metrics = self.agent.update()
                else:
                    metrics = {}

                # Optional: limit steps per episode
                if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                    done = True

            episode_rewards.append(ep_reward)

            if (ep + 1) % 10 == 0:
                avg_last_10 = np.mean(episode_rewards[-10:])
                print(
                    f"[Episode {ep+1}/{n_episodes}] "
                    f"total_steps={total_steps}, "
                    f"avg_reward(last10)={avg_last_10:.4f}, "
                    f"epsilon={self.agent.epsilon():.3f}"
                )

        print("[BuyTrainer] Training finished.")
        return episode_rewards

    def make_greedy_policy(self):
        """
        Returns a small function that runs the trained agent in greedy mode.
        """

        def policy(env: BuyEnv):
            state = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            while not done:
                action = self.agent.select_action(state, greedy=True)
                next_state, reward, done, info = env.step(action)
                state = next_state
                total_reward += reward
                steps += 1
            return total_reward, steps

        return policy
