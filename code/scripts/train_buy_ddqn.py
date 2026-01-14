import numpy as np
import matplotlib.pyplot as plt
import torch
import multiprocessing as mp

from config.loader import load_config
from agents.buy_agent_trainer import BuyAgentTrainer
from agents.multi_process.multi_process_trainer import MultiProcessTrainer
from agents.multi_process.handler import EnvHandler


def train_buy_agent_mp():
    # 0) Config
    config = load_config("config.yaml")  # adjust path if needed

    # 1) Build trainer (your NEW signature)
    buy_trainer = BuyAgentTrainer(
        ticker="AAPL",
        config=config,
        device="cpu",
    )

    assert buy_trainer.env is not None
    assert buy_trainer.agent is not None
    assert buy_trainer.state_df is not None
    assert buy_trainer.prices is not None

    print(f"[BuyTrainer-MP] state_df shape: {buy_trainer.state_df.shape}")
    print(f"[BuyTrainer-MP] prices shape: {buy_trainer.prices.shape}")

    # 2) Picklable env factory (uses NEW BuyEnv signature)
    env_fn = EnvHandler(
        env_type="buy",
        features=buy_trainer.state_df.values.astype(np.float32),
        prices=np.asarray(buy_trainer.prices, dtype=np.float32),
        config=config,
    )

    # 3) MP trainer (Option 2: shared q_net)
    mp_trainer = MultiProcessTrainer(
        agent=buy_trainer.agent,
        env_fn=env_fn,
        n_workers=config.training.n_workers if hasattr(config.training, "n_workers") else 4,
        steps_per_batch=config.training.steps_per_batch if hasattr(config.training, "steps_per_batch") else 300,
        worker_epsilon=0.05,  # you can tune this
    )

    print("[BuyTrainer-MP] Starting MP training...")

    mp_trainer.train(
        n_batches=600,
        updates_per_batch=50,
        log_every=10,
    )

    print(
        f"[BuyTrainer-MP] Done. "
        f"Replay buffer size={len(buy_trainer.agent.replay_buffer)} | "
        f"Loss entries={len(buy_trainer.agent.loss_history)}"
    )

    # 4) Greedy evaluation on full BuyEnv
    reward, steps = evaluate_greedy_buy(buy_trainer)
    print("\n[BuyTrainer-MP] Greedy evaluation:")
    print(f"Reward: {reward:.4f} | Steps: {steps}")

    # 5) Confidence distribution
    avg_buy, max_buy, min_buy = inspect_buy_confidence(buy_trainer)
    print("\n[BuyTrainer-MP] BUY confidence stats:")
    print(f"Average BUY confidence: {avg_buy:.6f}")
    print(f"Max BUY confidence:     {max_buy:.6f}")
    print(f"Min BUY confidence:     {min_buy:.6f}")

    # 6) Plot loss curve
    plot_loss_history(buy_trainer.agent)

    return buy_trainer


def evaluate_greedy_buy(buy_trainer):
    env = buy_trainer.env
    agent = buy_trainer.agent
    assert env is not None and agent is not None

    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action = agent.select_action(state, greedy=True)
        next_state, reward, done, info = env.step(action)
        total_reward += float(reward)
        state = next_state
        steps += 1

    return total_reward, steps


def inspect_buy_confidence(buy_trainer):
    agent = buy_trainer.agent
    state_df = buy_trainer.state_df
    assert agent is not None and state_df is not None

    confs = []
    agent.q_net.eval()

    for i in range(len(state_df)):
        state = state_df.iloc[i].values.astype(np.float32)
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(agent.device)
            q = agent.q_net(s)[0].detach().cpu().numpy()
            exps = np.exp(q - np.max(q))
            probs = exps / (np.sum(exps) + 1e-12)
            confs.append(float(probs[1]))  # index 1 = BUY

    confs = np.asarray(confs, dtype=np.float32)
    return float(confs.mean()), float(confs.max()), float(confs.min())


def plot_loss_history(agent):
    if not agent.loss_history:
        print("[BuyTrainer-MP] No loss history to plot.")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(agent.loss_history)
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.title("BuyAgent DDQN Loss (MP training)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # macOS must-have
    buy_trainer = train_buy_agent_mp()
