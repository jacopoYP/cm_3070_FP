from dataclasses import dataclass

@dataclass(frozen=True)
class AgentConfig:
    state_dim: int
    n_actions: int
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 200_000
    target_update_freq: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000
