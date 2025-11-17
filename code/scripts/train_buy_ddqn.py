# scripts/train_buy_ddqn.py

import os
import sys
import random
import numpy as np
import torch

# --- Ensure project root is on PYTHONPATH ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Now imports work
from agents.buy_agent_ddqn import BuyAgentTrainer


from agents.buy_agent_ddqn import BuyAgentTrainer

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # Just to be safe / reproducible
    set_seeds(42)

    # Make sure we run from project root
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    # 1) create trainer for one ticker (e.g. AAPL)
    trainer = BuyAgentTrainer(
        cfg_path="config/data_config.yaml",
        ticker="AAPL",
        window_size=30,
        horizon=5,
        transaction_cost=0.001,
    )

    # 2) quick training run (PoC)
    #    start small (e.g. 50 episodes) to make sure everything works
    rewards = trainer.train(
        n_episodes=50,
        max_steps_per_episode=None,  # let env end naturally
        warmup_steps=500,
    )

    print("Last 10 episode rewards:", rewards[-10:])

    # 3) run greedy policy once to see how it behaves
    greedy_policy = trainer.make_greedy_policy()
    total_reward, steps = greedy_policy(trainer.env)
    print(f"Greedy run: total_reward={total_reward:.4f}, steps={steps}")
