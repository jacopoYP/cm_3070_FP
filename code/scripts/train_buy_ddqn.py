import os
import sys
import random

import numpy as np
import torch

# --- Ensure project root is on PYTHONPATH ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from agents.buy_agent_ddqn import BuyAgentTrainer

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_seeds(42)

    # Run from project root
    os.chdir(ROOT)

    trainer = BuyAgentTrainer(
        cfg_path="config/data_config.yaml",
        ticker="AAPL",
        window_size=30,
        horizon=5,
        transaction_cost=0.001,
    )

    rewards = trainer.train(
        n_episodes=50,
        warmup_steps=500,
        max_steps_per_episode=None,
    )

    print("Last 10 episode rewards:", rewards[-10:])

    greedy_policy = trainer.make_greedy_policy()
    total_reward, steps = greedy_policy()
    print(f"Greedy run: total_reward={total_reward:.4f}, steps={steps}")
