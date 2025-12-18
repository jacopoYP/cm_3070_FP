import numpy as np
import matplotlib.pyplot as plt
import torch

from agents.buy_agent_trainer import BuyAgentTrainer
from agents.multi_process.multi_process_trainer import MultiProcessTrainer
from agents.multi_process.handler import EnvHandler, AgentHandler


def train_buy_agent_mp():
    # 1) Build trainer with your usual settings
    buy_trainer = BuyAgentTrainer(
        ticker="AAPL",
        window_size=30,
        horizon=20,
        transaction_cost=0.001,
        lambda_dd=0.05,
        lambda_vol=0.01,
        hold_penalty_long=0.0,
        device="cpu",
    )

    # 2) Ensure dataset + env + agent are built
    if buy_trainer.env is None or buy_trainer.agent is None:
        buy_trainer._build_dataset_and_env()

    print(f"[BuyTrainer-MP] state_df shape: {buy_trainer.state_df.shape}")
    print(f"[BuyTrainer-MP] prices shape: {buy_trainer.prices.shape}")

    # 3) Create EnvFactory and AgentFactory for MP
    EnvClass = buy_trainer.env.__class__

    env_fn = EnvHandler(
        state_df=buy_trainer.state_df,
        prices=buy_trainer.prices,
        EnvClass=EnvClass,
        horizon=buy_trainer.horizon,
        transaction_cost=buy_trainer.transaction_cost,
        # NOTE: Sell-specific kwargs are ignored for BuyEnv by EnvFactory
    )

    agent_fn = AgentHandler(buy_trainer.agent)

    # 4) Multi-process trainer
    mp_trainer = MultiProcessTrainer(
        agent=buy_trainer.agent,
        env_fn=env_fn,
        agent_fn=agent_fn,
        n_workers=4,          # you can go higher if CPU allows
        steps_per_batch=300,  # per worker per batch
    )

    print("[BuyTrainer-MP] Starting MP training...")

    # Recommended: 600–800 batches for a decent run
    mp_trainer.train(
        n_batches=600,
        updates_per_batch=50,
    )

    print(
        f"[BuyTrainer-MP] Done. "
        f"Replay buffer size={len(buy_trainer.agent.replay_buffer)} | "
        f"Loss entries={len(buy_trainer.agent.loss_history)}"
    )

    # 5) Greedy evaluation on full BuyEnv
    reward, steps = evaluate_greedy_buy(buy_trainer)
    print("\n[BuyTrainer-MP] Greedy evaluation:")
    print(f"Reward: {reward:.4f} | Steps: {steps}")

    # 6) Inspect BUY confidence distribution
    avg_buy, max_buy, min_buy = inspect_buy_confidence(buy_trainer)
    print("\n[BuyTrainer-MP] BUY confidence stats:")
    print(f"Average BUY confidence: {avg_buy:.6f}")
    print(f"Max BUY confidence:     {max_buy:.6f}")
    print(f"Min BUY confidence:     {min_buy:.6f}")

    # 7) Plot loss curve
    plot_loss_history(buy_trainer.agent)

    # Optionally: return trainer to reuse in TradeManager
    return buy_trainer


def evaluate_greedy_buy(buy_trainer):
    """
    Simple greedy rollout using the existing BuyEnv.
    """
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
        total_reward += reward
        state = next_state
        steps += 1

    return total_reward, steps


def inspect_buy_confidence(buy_trainer):
    """
    Run the trained BuyAgent over the entire state_df and
    compute BUY probabilities (softmax over Q-values).
    """
    agent = buy_trainer.agent
    state_df = buy_trainer.state_df
    assert agent is not None and state_df is not None

    confs = []
    for i in range(len(state_df)):
        state = state_df.iloc[i].values.astype(np.float32)
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(agent.device)
            q = agent.q_net(s)[0].cpu().numpy()
            exps = np.exp(q - np.max(q))
            probs = exps / np.sum(exps)
            confs.append(probs[1])  # index 1 = BUY

    confs = np.array(confs)
    return float(confs.mean()), float(confs.max()), float(confs.min())


def plot_loss_history(agent):
    """
    Plot the DDQN loss history.
    """
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
    buy_trainer = train_buy_agent_mp()
