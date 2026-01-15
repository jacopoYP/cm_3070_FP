# agents/multi_process/handler.py
# from agents.ddqn import DDQNAgent  # adjust import path if needed
import numpy as np

class EnvHandler:
    """
    Picklable env factory for macOS spawn.
    Works with your NEW BuyEnv / SellEnv signatures.
    """

    def __init__(self, env_type: str, features: np.ndarray, prices: np.ndarray, config, entry_indices=None):
        assert env_type in ("buy", "sell")
        self.env_type = env_type
        self.features = np.asarray(features, dtype=np.float32)
        self.prices = np.asarray(prices, dtype=np.float32)
        self.config = config
        self.entry_indices = None if entry_indices is None else np.asarray(entry_indices, dtype=int)

    def __call__(self):
        if self.env_type == "buy":
            from envs.buy_env import BuyEnv
            return BuyEnv(
                features=self.features,
                prices=self.prices,
                config=self.config,
            )

        # sell
        from envs.sell_env import SellEnv
        if self.entry_indices is None or len(self.entry_indices) == 0:
            raise ValueError("EnvHandler(sell) requires non-empty entry_indices")

        return SellEnv(
            state_df=self.features,
            prices=self.prices,
            entry_indices=self.entry_indices,
            config=self.config,
        )

class AgentHandler:
    def __init__(self, agent_cfg, device="cpu"):
        self.agent_cfg = agent_cfg
        self.device = device

    def __call__(self):
        from agents.ddqn import DDQNAgent
        return DDQNAgent(cfg=self.agent_cfg, device=self.device)



# class AgentHandler:
#     def __init__(self, agent_template: DDQNAgent):
#         self.template = agent_template

#     def __call__(self):
#         A = self.template

#         # Rebuild a fresh agent with the same hyperparameters
#         return DDQNAgent(
#             state_dim=A.state_dim,
#             n_actions=A.n_actions,
#             gamma=A.gamma,
#             lr=A.lr,
#             batch_size=A.batch_size,
#             buffer_size=A.buffer_size,
#             target_update_freq=A.target_update_freq,
#             epsilon_start=A.epsilon_start,
#             epsilon_end=A.epsilon_end,
#             epsilon_decay_steps=A.epsilon_decay_steps,
#             device=A.device.type,  # 'cpu' or 'cuda'
#         )

