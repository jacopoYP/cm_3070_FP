#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import sys
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---- your project imports (adjust paths if needed)
# from core.io import load_yaml_to_ns  # your existing helper
# from core.seeds import set_global_seeds  # your existing helper
from agents.ddqn_agent import DDQNAgent
from envs.sell_env import SellEnv  # only needed if you later extend; not used in v0
from trade.trade_manager import TradeManager


# -----------------------------
# Small utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_run_id(prefix: str) -> str:
    return f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

def save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def append_jsonl(path: str, obj) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")

def summarize_trades(trades: List[dict]) -> Dict[str, float]:
    if not trades:
        return {
            "n_trades": 0,
            "avg_net_return": 0.0,
            "win_rate": 0.0,
            "min_net": 0.0,
            "median_net": 0.0,
            "max_net": 0.0,
        }
    net = np.array([t["net_return"] for t in trades], dtype=float)
    return {
        "n_trades": int(len(net)),
        "avg_net_return": float(net.mean()),
        "win_rate": float((net > 0).mean()),
        "min_net": float(net.min()),
        "median_net": float(np.median(net)),
        "max_net": float(net.max()),
    }

def split_by_segments(
    features: np.ndarray,
    prices: np.ndarray,
    seg_len: int,
    train_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    features/prices are stacked by ticker segments of equal seg_len.
    Returns X_train, p_train, X_test, p_test and segment lens in those subsets.
    """
    seg_len = int(seg_len)
    train_frac = float(train_frac)

    n = len(prices)
    if n % seg_len != 0:
        raise ValueError(f"Expected stacked segments length multiple of seg_len. n={n}, seg_len={seg_len}")
    n_segs = n // seg_len
    train_len = int(seg_len * train_frac)
    test_len = seg_len - train_len

    train_idx = []
    test_idx = []
    for seg in range(n_segs):
        start = seg * seg_len
        train_idx.extend(range(start, start + train_len))
        test_idx.extend(range(start + train_len, start + seg_len))

    train_idx = np.array(train_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)

    X_train = features[train_idx]
    p_train = prices[train_idx]
    X_test = features[test_idx]
    p_test = prices[test_idx]

    return X_train, p_train, X_test, p_test, train_len, test_len, n_segs


# -----------------------------
# GA bits (simple)
# -----------------------------
@dataclass
class Individual:
    buy_min_confidence: float
    fitness: float = -1e18
    metrics: Dict = None

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def tournament_select(pop: List[Individual], k: int, rng: random.Random) -> Individual:
    contestants = rng.sample(pop, k=min(k, len(pop)))
    contestants.sort(key=lambda ind: ind.fitness, reverse=True)
    return contestants[0]

def mutate_buy_conf(x: float, sigma: float, lo: float, hi: float, rng: random.Random) -> float:
    return clamp(x + rng.gauss(0.0, sigma), lo, hi)

def make_initial_population(
    pop_size: int,
    lo: float,
    hi: float,
    rng: random.Random,
) -> List[Individual]:
    return [Individual(buy_min_confidence=rng.uniform(lo, hi)) for _ in range(pop_size)]


# -----------------------------
# Evaluation: TM BUY-ONLY
# -----------------------------
def run_tm_buy_only(
    cfg,
    state: np.ndarray,
    prices: np.ndarray,
    buy_agent: DDQNAgent,
    segment_len: int,
) -> Dict:
    tm = TradeManager(
        buy_agent=buy_agent,
        sell_agent=None,
        state=state,
        prices=prices,
        reward=cfg.reward,
        trade=cfg.trade_manager,
        segment_len=segment_len,
    )
    return tm.run()

def load_yaml_to_ns(path: str) -> SimpleNamespace:
    """
    Minimal YAML loader that returns a dot-accessible namespace.
    Avoids forcing OmegaConf; works with plain PyYAML.

    Example: cfg.reward.transaction_cost
    """
    import yaml  # local import to keep deps minimal

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    def rec(x):
        if isinstance(x, dict):
            return SimpleNamespace(**{k: rec(v) for k, v in x.items()})
        if isinstance(x, list):
            return [rec(v) for v in x]
        return x

    return rec(data)


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

def evaluate_individual(
    cfg_base,
    buy_agent: DDQNAgent,
    X_train: np.ndarray,
    p_train: np.ndarray,
    X_test: np.ndarray,
    p_test: np.ndarray,
    seg_train_len: int,
    seg_test_len: int,
    buy_min_conf: float,
) -> Tuple[float, Dict]:
    # clone cfg and override ONLY buy_min_confidence
    cfg = copy.deepcopy(cfg_base)
    cfg.trade_manager.buy_min_confidence = float(buy_min_conf)

    res_train = run_tm_buy_only(cfg, X_train, p_train, buy_agent, segment_len=seg_train_len)
    res_test  = run_tm_buy_only(cfg, X_test,  p_test,  buy_agent, segment_len=seg_test_len)

    train_stats = summarize_trades(res_train.get("trades", []))
    test_stats  = summarize_trades(res_test.get("trades", []))

    # simple fitness — focus on test
    fitness = float(res_test.get("final_equity", 0.0))

    metrics = {
        "buy_min_confidence": float(buy_min_conf),
        "train": {
            **train_stats,
            "final_equity": float(res_train.get("final_equity", 0.0)),
            "entry_debug": res_train.get("entry_debug", {}),
            "exit_reasons": res_train.get("exit_reasons", {}),
        },
        "test": {
            **test_stats,
            "final_equity": float(res_test.get("final_equity", 0.0)),
            "entry_debug": res_test.get("entry_debug", {}),
            "exit_reasons": res_test.get("exit_reasons", {}),
        },
    }
    return fitness, metrics


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--features", type=str, default="data/features.npy")
    ap.add_argument("--prices", type=str, default="data/prices.npy")
    ap.add_argument("--buy_model", type=str, required=True, help="Path to trained buy_agent.pt to reuse")
    ap.add_argument("--out_root", type=str, default="runs_ga")
    ap.add_argument("--seed", type=int, default=42)

    # GA params
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--tourn", type=int, default=3)
    ap.add_argument("--mut_sigma", type=float, default=0.03)
    ap.add_argument("--lo", type=float, default=0.35)
    ap.add_argument("--hi", type=float, default=0.85)

    args = ap.parse_args()

    cfg = load_yaml_to_ns(args.config)
    set_global_seeds(int(args.seed))
    rng = random.Random(int(args.seed))

    run_id = now_run_id("ga_buy_conf")
    out_dir = os.path.join(args.out_root, run_id)
    ensure_dir(out_dir)

    # data
    features = np.load(args.features).astype(np.float32, copy=False)
    prices = np.load(args.prices).astype(np.float32, copy=False)

    seg_len = int(getattr(cfg.data, "seg_len", 1239))
    train_frac = float(getattr(cfg.data, "train_frac", 0.7))
    X_train, p_train, X_test, p_test, seg_train_len, seg_test_len, n_segs = split_by_segments(
        features, prices, seg_len=seg_len, train_frac=train_frac
    )

    # load BUY agent once (GA reuses it)
    buy_cfg = copy.deepcopy(cfg.agent)
    buy_cfg.state_dim = int(X_train.shape[1])
    buy_cfg.n_actions = 2
    buy_agent = DDQNAgent(buy_cfg)
    buy_agent.load(args.buy_model)

    # logs
    jsonl_path = os.path.join(out_dir, "ga_log.jsonl")
    meta_path = os.path.join(out_dir, "meta.json")
    save_json(meta_path, {
        "run_id": run_id,
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "data": {
            "features": args.features,
            "prices": args.prices,
            "seg_len": seg_len,
            "n_segs": n_segs,
            "train_frac": train_frac,
            "seg_train_len": seg_train_len,
            "seg_test_len": seg_test_len,
        },
        "ga": {
            "pop": args.pop,
            "gens": args.gens,
            "elite": args.elite,
            "tourn": args.tourn,
            "mut_sigma": args.mut_sigma,
            "lo": args.lo,
            "hi": args.hi,
        },
        "buy_model": args.buy_model,
        "fitness": "test.final_equity (TM buy-only)",
    })

    # init pop
    pop = make_initial_population(args.pop, args.lo, args.hi, rng)

    best_overall: Individual | None = None

    for gen in range(args.gens):
        # evaluate
        for i, ind in enumerate(pop):
            fit, metrics = evaluate_individual(
                cfg_base=cfg,
                buy_agent=buy_agent,
                X_train=X_train,
                p_train=p_train,
                X_test=X_test,
                p_test=p_test,
                seg_train_len=seg_train_len,
                seg_test_len=seg_test_len,
                buy_min_conf=ind.buy_min_confidence,
            )
            ind.fitness = float(fit)
            ind.metrics = metrics

            append_jsonl(jsonl_path, {
                "gen": gen,
                "idx": i,
                "buy_min_confidence": ind.buy_min_confidence,
                "fitness": ind.fitness,
                "train": {
                    "final_equity": metrics["train"]["final_equity"],
                    "n_trades": metrics["train"]["n_trades"],
                    "avg_net_return": metrics["train"]["avg_net_return"],
                    "win_rate": metrics["train"]["win_rate"],
                },
                "test": {
                    "final_equity": metrics["test"]["final_equity"],
                    "n_trades": metrics["test"]["n_trades"],
                    "avg_net_return": metrics["test"]["avg_net_return"],
                    "win_rate": metrics["test"]["win_rate"],
                },
            })

        # sort by fitness
        pop.sort(key=lambda ind: ind.fitness, reverse=True)

        if best_overall is None or pop[0].fitness > best_overall.fitness:
            best_overall = copy.deepcopy(pop[0])

        # console snapshot
        print(f"\n=== GEN {gen} ===")
        print("BEST:", pop[0].buy_min_confidence, "fitness:", pop[0].fitness)
        print("TEST:", pop[0].metrics["test"]["final_equity"],
              "n_trades:", pop[0].metrics["test"]["n_trades"],
              "avg_net:", pop[0].metrics["test"]["avg_net_return"],
              "win_rate:", pop[0].metrics["test"]["win_rate"])

        # next generation
        elite_n = max(0, min(args.elite, len(pop)))
        next_pop = [copy.deepcopy(pop[j]) for j in range(elite_n)]

        while len(next_pop) < args.pop:
            parent = tournament_select(pop, args.tourn, rng)
            child_conf = mutate_buy_conf(parent.buy_min_confidence, args.mut_sigma, args.lo, args.hi, rng)
            next_pop.append(Individual(buy_min_confidence=child_conf))

        pop = next_pop

    # final save best
    best_path = os.path.join(out_dir, "best.json")
    save_json(best_path, {
        "best_buy_min_confidence": best_overall.buy_min_confidence if best_overall else None,
        "best_fitness": best_overall.fitness if best_overall else None,
        "best_metrics": best_overall.metrics if best_overall else None,
        "log_jsonl": jsonl_path,
        "meta": meta_path,
    })

    print("\n=== GA DONE ===")
    print("Best buy_min_confidence:", best_overall.buy_min_confidence)
    print("Best fitness (test final_equity):", best_overall.fitness)
    print("Saved:", best_path)
    print("Log:", jsonl_path)


if __name__ == "__main__":
    main()
