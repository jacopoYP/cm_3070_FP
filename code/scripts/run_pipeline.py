from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
# from typing import Any, Dict, List, Optional, Tuple
from typing import Any
import logging

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.helper import split_by_segments, load_yaml_to_ns, now_run_id, check_dir
from core.logging_config import setup_logging
from core.metrics import summarize_trades

from agents.ddqn_agent import DDQNAgent
from core.types import AgentConfig
from envs.buy_env import BuyEnv
from envs.sell_env import SellEnv
from trade.trade_manager import TradeManager


# ---------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------
# TODO: Clean
# def now_run_id(prefix: str = "pipeline") -> str:
#     return f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# def check_dir(path: str) -> None:
#     os.makedirs(path, exist_ok=True)


def json_default(o: Any):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=json_default)

def namespace_to_dict(ns):
    if isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    if hasattr(ns, "__dict__"):
        return {k: namespace_to_dict(v) for k, v in ns.__dict__.items()}
    if isinstance(ns, list):
        return [namespace_to_dict(x) for x in ns]
    return ns


def build_agent_cfg(cfg: SimpleNamespace, state_dim: int, n_actions: int) -> "AgentConfig":
    d = namespace_to_dict(cfg.agent)
    d["state_dim"] = int(state_dim)
    d["n_actions"] = int(n_actions)
    return AgentConfig(**d)

def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

# TODO: Clean
# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
# def summarize_trades(trades: List[Dict[str, Any]]) -> Dict[str, float]:
#     if not trades:
#         return {
#             "n_trades": 0,
#             "avg_net_return": 0.0,
#             "win_rate": 0.0,
#             "min_net": 0.0,
#             "median_net": 0.0,
#             "max_net": 0.0,
#         }

#     net = np.array([t.get("net_return", 0.0) for t in trades], dtype=float)
#     return {
#         "n_trades": int(len(trades)),
#         "avg_net_return": float(np.mean(net)),
#         "win_rate": float(np.mean(net > 0)),
#         "min_net": float(np.min(net)),
#         "median_net": float(np.median(net)),
#         "max_net": float(np.max(net)),
#     }


# ---------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------
def train_buy_agent(
    cfg: SimpleNamespace,
    X_train: np.ndarray,
    p_train: np.ndarray,
    out_dir: str,
) -> str:
    buy_env = BuyEnv(
        features=X_train,
        prices=p_train,
        reward=cfg.reward,
        trade=cfg.trade_manager,
    )

    tmp_env = BuyEnv(features=X_train, prices=p_train, reward=cfg.reward, trade=cfg.trade_manager)
    buy_cfg = build_agent_cfg(cfg, state_dim=int(tmp_env.state_dim), n_actions=2)
    buy_agent = DDQNAgent(buy_cfg)

    EPISODES = int(getattr(cfg.training, "buy_episodes", 200))
    MAX_STEPS = int(getattr(cfg.training, "buy_max_steps", 10_000_000))
    UPDATES_PER_STEP = int(getattr(cfg.training, "buy_updates_per_step", 1))
    WARMUP = int(getattr(cfg.training, "warmup_steps", 200))

    # Looping episodes
    for ep in range(EPISODES):
        # Starting with reset, the same way was done in the Atari games (e.g. Lunar)
        s = buy_env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while (not done) and (steps < MAX_STEPS):
            action = buy_agent.select_action(s, greedy=False)
            ns, r, done, info = buy_env.step(action)

            buy_agent.push(s, action, r, ns, done)

            if buy_agent.total_steps >= WARMUP:
                for _ in range(UPDATES_PER_STEP):
                    buy_agent.update()

            s = ns
            ep_reward += float(r)
            steps += 1

        if (ep + 1) % int(getattr(cfg.training, "log_every", 10)) == 0:
            loss = buy_agent.loss_history[-1] if buy_agent.loss_history else None
            print(f"[BUY] ep={ep+1}/{EPISODES} reward={ep_reward:.4f} eps={buy_agent.eps:.3f} loss={loss}")

    path = os.path.join(out_dir, "buy_agent.pt")
    buy_agent.save(path)
    return path

# ---------------------------------------------------------------------
# Trade Manager
# ---------------------------------------------------------------------
def run_tm(
    cfg,
    state: np.ndarray,
    prices: np.ndarray,
    buy_agent,
    segment_len: int,
    name: str,
    sell_agent=None,
) -> dict:
    tm = TradeManager(
        buy_agent=buy_agent,
        sell_agent=sell_agent,
        state=state,
        prices=prices,
        reward=cfg.reward,
        trade=cfg.trade_manager,
        segment_len=segment_len,
    )
    res = tm.run()

    trades = res.get("trades", [])
    stats = summarize_trades(trades)

    label = "WITH SELL" if sell_agent is not None else "BUY ONLY"
    print(f"\n=== TM {label} ({name}) ===")
    print("n_trades:", stats["n_trades"])
    print("final_equity:", float(res.get("final_equity", 0.0)))
    print("avg_net_return:", stats["avg_net_return"], "win_rate:", stats["win_rate"])
    print("min/median/max net:", stats["min_net"], stats["median_net"], stats["max_net"])
    print("exit_reasons:", res.get("exit_reasons", {}))
    print("entry_debug:", {k: res.get("entry_debug", {}).get(k) for k in ["checked","opened","blocked_sentiment","blocked_conf","blocked_trend","blocked_latest_entry"]})
    print("sell_debug:", res.get("sell_debug", {}))

    return {
        "final_equity": float(res.get("final_equity", 0.0)),
        "trades": trades,
        "exit_reasons": res.get("exit_reasons", {}),
        "entry_debug": res.get("entry_debug", {}),
        "sell_debug": res.get("sell_debug", {}),
        **stats,
    }

# TODO: Remove
# def uplift(a: dict, b: dict) -> dict:
#     return {
#         "final_equity_delta": float(a.get("final_equity", 0.0) - b.get("final_equity", 0.0)),
#         "avg_net_return_delta": float(a.get("avg_net_return", 0.0) - b.get("avg_net_return", 0.0)),
#         "win_rate_delta": float(a.get("win_rate", 0.0) - b.get("win_rate", 0.0)),
#         "n_trades_delta": int(a.get("n_trades", 0) - b.get("n_trades", 0)),
#         "sell_exits": int(a.get("exit_reasons", {}).get("sell_agent", 0)),
#     }


def collect_entries_for_sell_training(
    cfg: SimpleNamespace,
    state: np.ndarray,
    prices: np.ndarray,
    buy_agent: DDQNAgent,
    segment_len: int,
    out_path: str,
) -> np.ndarray:
    tm = TradeManager(
        buy_agent=buy_agent,
        sell_agent=None,
        state=state,
        prices=prices,
        reward=cfg.reward,
        trade=cfg.trade_manager,
        segment_len=segment_len,
    )
    # TODO: Clean
    # if hasattr(tm, "collect_entry_indices_topk"):
    topk = int(getattr(cfg.training, "sell_topk_entries_per_segment", 50))
    min_gap = getattr(cfg.training, "sell_min_gap", None)
    use_conf = bool(getattr(cfg.training, "sell_use_confidence_score", False))
    entries = tm.collect_entry_indices_topk(
        topk_per_segment=topk,
        min_gap=min_gap,
        use_confidence_score=use_conf,
    )
    entries = np.asarray(entries, dtype=np.int64)
    # else:
    #     # fallback (less ideal): run buy-only TM and grab entries
    #     out = tm.run()
    #     entries = np.asarray(out.get("entry_indices", []), dtype=np.int64)

    np.save(out_path, entries)
    return entries


def train_sell_agent(
    cfg: SimpleNamespace,
    X_train: np.ndarray,
    p_train: np.ndarray,
    entries_train: np.ndarray,
    seg_train_len: int,
    out_dir: str,
) -> str:
    sell_env_train = SellEnv(
        features=X_train,
        prices=p_train,
        entry_indices=entries_train,
        transaction_cost=cfg.reward.transaction_cost,
        sell_horizon=cfg.trade_manager.sell_horizon,
        min_hold_bars=cfg.trade_manager.min_hold_bars,
        segment_len=seg_train_len,
        include_pos_features=True,
    )

    sell_cfg = build_agent_cfg(cfg, state_dim=int(sell_env_train.state_dim), n_actions=2)
    sell_cfg.lr = float(getattr(getattr(cfg.training, "sell_overrides", SimpleNamespace()), "lr", sell_cfg.lr))

    # DEfault params in case
    EPISODES = int(getattr(cfg.training, "sell_episodes", 4000))
    MAX_STEPS = int(getattr(cfg.training, "sell_max_steps", 200))
    UPDATES_PER_STEP = int(getattr(cfg.training, "sell_updates_per_step", 1))
    WARMUP = int(getattr(cfg.training, "warmup_steps", 200))

    # Epsilon decay heuristic
    avg_steps_per_ep = int(getattr(cfg.training, "sell_avg_steps_per_ep", 15))
    total_estimated_steps = EPISODES * avg_steps_per_ep
    sell_cfg.epsilon_start = float(getattr(sell_cfg, "epsilon_start", 1.0))
    sell_cfg.epsilon_end = float(getattr(sell_cfg, "epsilon_end", 0.05))
    sell_cfg.epsilon_decay_steps = int(getattr(cfg.training, "sell_decay_steps", int(total_estimated_steps * 0.8)))

    sell_agent = DDQNAgent(sell_cfg)
    print(f"SELL state_dim={sell_cfg.state_dim} decay_steps={sell_cfg.epsilon_decay_steps} est_total={total_estimated_steps}")

    for ep in range(EPISODES):
        s = sell_env_train.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while (not done) and (steps < MAX_STEPS):
            action = sell_agent.select_action(s, greedy=False)
            ns, r, done, info = sell_env_train.step(action)

            sell_agent.push(s, action, r, ns, done)

            if sell_agent.total_steps >= WARMUP:
                for _ in range(UPDATES_PER_STEP):
                    sell_agent.update()

            s = ns
            ep_reward += float(r)
            steps += 1

        if (ep + 1) % int(getattr(cfg.training, "log_every", 10)) == 0:
            loss = sell_agent.loss_history[-1] if sell_agent.loss_history else None
            print(f"[SELL] ep={ep+1}/{EPISODES} reward={ep_reward:.4f} eps={sell_agent.eps:.3f} loss={loss}")

    path = os.path.join(out_dir, "sell_agent.pt")
    sell_agent.save(path)
    return path

# def load_agent_for_eval(agent_cls, cfg_agent: SimpleNamespace, model_path: str):
#     agent = agent_cls(cfg_agent)
#     agent.load(model_path)
#     return agent

# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--features", type=str, default="data/features.npy")
    ap.add_argument("--prices", type=str, default="data/prices.npy")
    ap.add_argument("--out_root", type=str, default="runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument("--meta", type=str, default="data/row_meta.parquet")

    # knobs
    ap.add_argument("--skip_buy_train", action="store_true")
    ap.add_argument("--skip_sell_train", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml_to_ns(args.config)
    set_global_seeds(int(args.seed))

    run_id = args.run_id or now_run_id("pipeline")
    out_dir = os.path.join(args.out_root, run_id)
    check_dir(out_dir)

    # Loading data
    features = np.load(args.features).astype(np.float32, copy=False)
    prices = np.load(args.prices).astype(np.float32, copy=False)

    train_frac = float(getattr(cfg.data, "train_frac", 0.7))

    X_train, p_train, X_test, p_test, seg_train_len, seg_test_len, n_segs = split_by_segments(
        features, prices, train_frac=train_frac, args=args, cfg=cfg
    )

    print("\n=== DATA SPLIT ===")
    print("features:", features.shape, "prices:", prices.shape)
    print("train_len per seg:", seg_train_len, "test_len per seg:", seg_test_len)
    print("X_train:", X_train.shape, "p_train:", p_train.shape)
    print("X_test :", X_test.shape, "p_test :", p_test.shape)

    # Training the buy agent
    buy_path = os.path.join(out_dir, "buy_agent.pt")
    if not args.skip_buy_train:
        buy_path = train_buy_agent(cfg, X_train, p_train, out_dir)
        print("Saved BUY:", buy_path)
    else:
        print("Skipping BUY training; expecting:", buy_path)

    # Recreate buy_agent for TM usage (clean instance)
    buy_cfg = deepcopy(cfg.agent)
    buy_cfg.state_dim = int(X_train.shape[1])
    buy_cfg.n_actions = 2
    buy_agent = DDQNAgent(buy_cfg)
    buy_agent.load(buy_path)

    # Collect entry indices (for sell training)
    entries_train_path = os.path.join(out_dir, "entry_indices_train_sell.npy")
    entries_test_path = os.path.join(out_dir, "entry_indices_test_sell.npy")

    entries_train = collect_entries_for_sell_training(
        cfg=cfg,
        state=X_train,
        prices=p_train,
        buy_agent=buy_agent,
        segment_len=seg_train_len,
        out_path=entries_train_path,
    )
    entries_test = collect_entries_for_sell_training(
        cfg=cfg,
        state=X_test,
        prices=p_test,
        buy_agent=buy_agent,
        segment_len=seg_test_len,
        out_path=entries_test_path,
    )

    print("entries_train:", entries_train.shape, "entries_test:", entries_test.shape)

    # Training now the sell agent
    sell_path = os.path.join(out_dir, "sell_agent.pt")
    if not args.skip_sell_train:
        sell_path = train_sell_agent(cfg, X_train, p_train, entries_train, seg_train_len, out_dir)
        print("Saved SELL:", sell_path)
    else:
        print("Skipping SELL training; expecting:", sell_path)

    # Recreate sell agent for TM usage
    sell_cfg = deepcopy(cfg.agent)
    tmp_sell_env = SellEnv(
        features=X_train,
        prices=p_train,
        entry_indices=entries_train,
        transaction_cost=cfg.reward.transaction_cost,
        sell_horizon=cfg.trade_manager.sell_horizon,
        min_hold_bars=cfg.trade_manager.min_hold_bars,
        segment_len=seg_train_len,
        include_pos_features=True,
    )
    sell_cfg.state_dim = int(tmp_sell_env.state_dim)
    sell_cfg.n_actions = 2
    sell_agent = DDQNAgent(sell_cfg)
    sell_agent.load(sell_path)

    # ---------------------------------------------------------------------
    # TRADE MANAGER BACKTESTS: BUY-ONLY (baseline) vs WITH-SELL
    # ---------------------------------------------------------------------

    # BUY ONLY (sell_agent=None)
    tm_train_buy_only = run_tm(
        cfg=cfg,
        state=X_train,
        prices=p_train,
        buy_agent=buy_agent,
        sell_agent=None,
        segment_len=seg_train_len,
        name="TRAIN",
    )

    tm_test_buy_only = run_tm(
        cfg=cfg,
        state=X_test,
        prices=p_test,
        buy_agent=buy_agent,
        sell_agent=None,
        segment_len=seg_test_len,
        name="TEST",
    )

    # WITH SELL
    tm_train_with_sell = run_tm(
        cfg=cfg,
        state=X_train,
        prices=p_train,
        buy_agent=buy_agent,
        sell_agent=sell_agent,
        segment_len=seg_train_len,
        name="TRAIN",
    )
    tm_test_with_sell = run_tm(
        cfg=cfg,
        state=X_test,
        prices=p_test,
        buy_agent=buy_agent,
        sell_agent=sell_agent,
        segment_len=seg_test_len,
        name="TEST",
    )

    # Save trades
    train_trades_buy_only_path = os.path.join(out_dir, "trades_train_buy_only.json")
    test_trades_buy_only_path  = os.path.join(out_dir, "trades_test_buy_only.json")
    train_trades_with_sell_path = os.path.join(out_dir, "trades_train_with_sell.json")
    test_trades_with_sell_path  = os.path.join(out_dir, "trades_test_with_sell.json")

    save_json(train_trades_buy_only_path, tm_train_buy_only.get("trades", []))
    save_json(test_trades_buy_only_path,  tm_test_buy_only.get("trades", []))
    save_json(train_trades_with_sell_path, tm_train_with_sell.get("trades", []))
    save_json(test_trades_with_sell_path,  tm_test_with_sell.get("trades", []))

    # Summary.json for debug purposes
    summary = {
        "run_id": run_id,
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "data": {
            "features_path": args.features,
            "prices_path": args.prices,
            "shape_features": list(features.shape),
            "shape_prices": list(prices.shape),
            # "seg_len": int(seg_len),
            "n_segs": int(n_segs),
            "train_frac": float(train_frac),
            "seg_train_len": int(seg_train_len),
            "seg_test_len": int(seg_test_len),
        },
        "train": {
            "buy": {
                "episodes": int(getattr(cfg.training, "buy_episodes", 200)),
                "steps_per_episode": None,
                "state_dim": int(X_train.shape[1]),
                "reward_mean": float(np.mean(getattr(buy_agent, "episode_rewards", [0.0]))),
                "reward_last": float((getattr(buy_agent, "episode_rewards", [0.0]) or [0.0])[-1]),
                "loss_last": float((getattr(buy_agent, "loss_history", [0.0]) or [0.0])[-1]),
                "total_steps": int(getattr(buy_agent, "total_steps", 0)),
                "learn_steps": int(getattr(buy_agent, "learn_steps", 0)),
                "model_path": buy_path,
            },
            "sell": {
                "episodes": int(getattr(cfg.training, "sell_episodes", 4000)),
                "max_steps": int(getattr(cfg.training, "sell_max_steps", 200)),
                "updates_per_step": int(getattr(cfg.training, "sell_updates_per_step", 1)),
                "avg_steps_per_ep": int(getattr(cfg.training, "sell_avg_steps_per_ep", 15)),
                "sell_state_dim": int(sell_cfg.state_dim),
                "sell_decay_steps": int(getattr(sell_cfg, "epsilon_decay_steps", 0)),
                "reward_mean": float(np.mean(getattr(sell_agent, "episode_rewards", [0.0]))),
                "reward_last": float((getattr(sell_agent, "episode_rewards", [0.0]) or [0.0])[-1]),
                "loss_last": float((getattr(sell_agent, "loss_history", [0.0]) or [0.0])[-1]),
                "model_path": sell_path,
                "n_entries_train": int(len(entries_train)),
            },
            "tm_buy_only": {k: v for k, v in tm_train_buy_only.items() if k != "trades"},
            "tm_with_sell": {k: v for k, v in tm_train_with_sell.items() if k != "trades"},
        },
        "test": {
            "tm_buy_only": {k: v for k, v in tm_test_buy_only.items() if k != "trades"},
            "tm_with_sell": {k: v for k, v in tm_test_with_sell.items() if k != "trades"},
        },
        # "sell_uplift": {
        #     "train": uplift(tm_train_with_sell, tm_train_buy_only),
        #     "test": uplift(tm_test_with_sell, tm_test_buy_only),
        # },
        "entry_indices": {
            "n_train": int(len(entries_train)),
            "n_test": int(len(entries_test)),
            "train_path": entries_train_path,
            "test_path": entries_test_path,
        },
        "artifacts": {
            "buy_agent": buy_path,
            "sell_agent": sell_path,
            "entry_indices_train_sell": entries_train_path,
            "entry_indices_test_sell": entries_test_path,
            "trades_train_buy_only": train_trades_buy_only_path,
            "trades_test_buy_only": test_trades_buy_only_path,
            "trades_train_with_sell": train_trades_with_sell_path,
            "trades_test_with_sell": test_trades_with_sell_path,
        },
    }

    summary_path = os.path.join(out_dir, "summary.json")
    save_json(summary_path, summary)

    print("\n=== SELL UPLIFT (WITH_SELL - BUY_ONLY) ===")
    print("TRAIN:", summary["sell_uplift"]["train"])
    print("TEST :", summary["sell_uplift"]["test"])

    print("\nSaved artifacts:")
    for k, v in summary["artifacts"].items():
        print(" -", v)
    print(" -", summary_path)

if __name__ == "__main__":
    main()