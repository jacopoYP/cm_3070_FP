import yaml
import numpy as np

from agents.ddqn import DDQNAgent
from envs.sell_env import SellEnv
from pipeline.build_dataset import make_state_frame
from features.state_assembler import StateAssembler
from config.system import TradingSystemConfig

from dataclasses import replace

class SellAgentTrainer:
    """
    Sell agent trainer.

    - Builds dataset for one ticker
    - Builds rolling-window states (shared state representation with BuyAgent)
    - Wraps everything in SellEnv
    - Trains a DDQN agent on HOLD vs SELL
    """

    def __init__(
        self,
        ticker: str,
        config: TradingSystemConfig,
        device: str | None = None,
    ):
        self.ticker = ticker
        self.config = config
        self.device = device

        self.env: SellEnv | None = None
        self.agent: DDQNAgent | None = None

        self.state_df: np.ndarray | None = None
        self.prices: np.ndarray | None = None

    # --------------------------------------------------
    # Dataset + Env Construction
    # --------------------------------------------------

    def _build_dataset_and_env(self, buy_entry_indices: list[int] | None = None):
        """
        1) Build base dataset (technicals + price) using existing pipeline
        2) Build rolling-window states with StateAssembler (window_size from config.state)
        3) Align prices
        4) Create SellEnv
        5) Create DDQN agent (hyperparams from config.agent)
        """
        dataset = make_state_frame(self.ticker, self.config)
        print(f"[SellTrainer] Raw dataset: {dataset.shape}")

        dataset = dataset.dropna()
        print(f"[SellTrainer] After dropna: {dataset.shape}")

        feature_cols = [c for c in dataset.columns if c != "price"]

        # window_size belongs to STATE construction, shared across Buy/Sell
        window_size = self.config.state.window_size

        assembler = StateAssembler(feature_cols=feature_cols, window_size=window_size)

        # assembler.assemble returns a DF whose index is aligned to the last rows after rolling window
        state_df = assembler.assemble(dataset)
        prices = dataset.loc[state_df.index, "price"].values.astype(np.float32)

        # Convert states to numpy (SellEnv expects ndarray)
        self.state_df = state_df.values.astype(np.float32)
        self.prices = prices

        print(f"[SellTrainer] state_df shape: {self.state_df.shape}")
        print(f"[SellTrainer] prices shape: {self.prices.shape}")

        # Sell horizon + costs belong to trade_manager
        horizon = self.config.trade_manager.sell_horizon
        # cost = self.config.trade_manager.transaction_cost
        cost = self.config.reward.transaction_cost

        # self.env = SellEnv(
        #     state_df=self.state_df,
        #     prices=self.prices,
        #     # horizon=horizon,
        #     # transaction_cost=cost,
        #     # Optional: enforce minimum holding period (if your SellEnv supports it)
        #     min_steps_before_sell=getattr(self.config.trade_manager, "min_steps_before_sell", 1),
        #     # Optional: restrict entry points to those produced by BuyAgent
        #     buy_entry_indices=buy_entry_indices,
        # )
        # -----------------------
        # Build Sell environment (CORRECT)
        # -----------------------
        buy_entry_indices = np.asarray(buy_entry_indices, dtype=int)
        if buy_entry_indices.ndim != 1 or len(buy_entry_indices) == 0:
            raise ValueError(
                "SellAgentTrainer: buy_entry_indices must be a non-empty 1D array."
            )


        self.env = SellEnv(
            state_df=self.state_df,
            prices=self.prices,
            entry_indices=buy_entry_indices,
            config=self.config,
        )

        # Agent hyperparams come from config.agent
        state_dim = self.state_df.shape[1]
        # self.agent = DDQNAgent(
        #     state_dim=state_dim,
        #     n_actions=2,  # HOLD / SELL
        #     gamma=self.config.agent.gamma,
        #     lr=self.config.agent.lr,
        #     batch_size=self.config.agent.batch_size,
        #     buffer_size=self.config.agent.buffer_size,
        #     target_update_freq=self.config.agent.target_update_freq,
        #     epsilon_start=self.config.agent.epsilon_start,
        #     epsilon_end=self.config.agent.epsilon_end,
        #     epsilon_decay_steps=self.config.agent.epsilon_decay_steps,
        #     device=self.device,
        # )
        # Inject runtime dimensions into agent config
        agent_cfg = replace(
            self.config.agent,
            state_dim=state_dim,
            n_actions=2,   # HOLD / SELL
        )

        self.agent = DDQNAgent(
            cfg=agent_cfg,
            device=self.device,
        )


        print(f"[SellTrainer] state_dim={state_dim}, actions=2")

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def train(
        self,
        n_episodes: int = 50,
        warmup_dynamic: bool = True,
        verbose: bool = True,
        buy_entry_indices: list[int] | None = None,
    ):
        """
        Train the SellAgent.

        - warmup_dynamic: warmup_steps = max(200, int(0.2 * len(state_df)))
        """
        # SAfety check
        if buy_entry_indices is None:
            raise ValueError(
                "SellAgentTrainer.train() requires buy_entry_indices.\n"
                "You must pass the BUY entry indices generated by BuyAgent."
            )

        if self.env is None or self.agent is None:
            self._build_dataset_and_env(buy_entry_indices=buy_entry_indices)

        assert self.env is not None
        assert self.agent is not None

        # Warmup
        if warmup_dynamic:
            warmup_steps = max(200, int(len(self.state_df) * 0.2))
        else:
            warmup_steps = 500

        if verbose:
            print(f"[SellTrainer] Warmup set to: {warmup_steps}")

        episode_rewards: list[float] = []

        for ep in range(1, n_episodes + 1):
            state = self.env.reset()
            done = False
            ep_reward = 0.0
            steps = 0

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                # ✅ your DDQNAgent expects push_transition (as you re-added)
                self.agent.push_transition(state, action, reward, next_state, done)

                # update after warmup
                if self.agent.learn_step > warmup_steps:
                    self.agent.update()

                self.agent.learn_step += 1

                state = next_state
                ep_reward += float(reward)
                steps += 1

            episode_rewards.append(ep_reward)

            if verbose and (ep == 1 or ep % 10 == 0):
                print(
                    f"[Sell Ep {ep}/{n_episodes}] "
                    f"Reward={ep_reward:.4f} | Steps={steps} | "
                    f"Eps={self.agent.epsilon:.3f} | Buffer={len(self.agent.replay_buffer)}"
                )

        if verbose:
            print("SellAgent Training COMPLETE.")
            print("Last 5 rewards:", episode_rewards[-5:])

        return episode_rewards
