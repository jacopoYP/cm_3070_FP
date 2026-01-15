from __future__ import annotations

import argparse
import os
import time

import numpy as np

from core.config import load_config
# from ..core.config import load_config
from agents.ddqn_agent import DDQNAgent
from envs.buy_env import BuyEnv
from diagnostics.q_gap import compute_q_gap, plot_q_gap

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--features_npy", required=True, help="Path to features array (n_steps,state_dim)")
    p.add_argument("--prices_npy", required=True, help="Path to prices array (n_steps,)")
    p.add_argument("--out_dir", default="runs")
    args = p.parse_args()

    cfg = load_config(args.config)

    features = np.load(args.features_npy)
    prices = np.load(args.prices_npy)

    cfg.agent.state_dim = int(features.shape[1])
    cfg.agent.n_actions = 2

    env = BuyEnv(features, prices, cfg.reward, cfg.trade_manager)
    agent = DDQNAgent(cfg.agent)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # training loop
    for ep in range(int(cfg.training.episodes)):
        s = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            a = agent.select_action(s, greedy=False)
            ns, r, done, info = env.step(a)
            agent.push(s, a, r, ns, done)

            if agent.learn_steps >= int(cfg.training.warmup_steps):
                for _ in range(1):
                    agent.update()

            s = ns
            ep_reward += float(r)
            steps += 1
            if cfg.training.steps_per_episode is not None and steps >= int(cfg.training.steps_per_episode):
                break

        if (ep + 1) % int(cfg.training.log_every) == 0:
            loss = agent.loss_history[-1] if agent.loss_history else None
            print(f"[BUY] ep={ep+1}/{cfg.training.episodes} reward={ep_reward:.4f} eps={agent.eps:.3f} loss={loss}")

    # save model + diagnostics
    model_path = os.path.join(out_dir, "buy_agent.pt")
    agent.save(model_path)

    gaps = compute_q_gap(agent, features, max_points=2000)
    paths = plot_q_gap(gaps, out_dir, tag="buy")

    print("Saved model:", model_path)
    print("Diagnostics:", paths)


if __name__ == "__main__":
    main()
