# scripts/run_ga.py
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import yaml

from ga.genome import GeneSpec, Genome
from ga.population import GAConfig, Population

from agents.ddqn_agent import DDQNAgent
from trade.trade_manager import TradeManager

from core.helper import split_by_ticker_time_ga
from envs.buy_env import BuyEnv

# ----------------------------
# utils
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def now_id(prefix: str) -> str:
    return f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

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

def set_by_path(d: Dict[str, Any], path: str, value: Any) -> None:
    """
    Sets d["a"]["b"]["c"] = value for path "a.b.c".
    Creates dicts if missing.
    """
    parts = path.split(".")
    cur = d
    for k in parts[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


def get_by_path(d: Dict[str, Any], path: str, default=None):
    parts = path.split(".")
    cur = d
    for k in parts:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def summarize_trades(trades: list[dict]) -> Dict[str, Any]:
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


# ----------------------------
# evaluation (TM WITH SELL)
# ----------------------------
def run_tm_with_sell(cfgd: Dict[str, Any], state: np.ndarray, prices: np.ndarray, buy_agent, sell_agent, segment_len: int):
    tm = TradeManager(
        buy_agent=buy_agent,
        sell_agent=sell_agent,
        state=state,
        prices=prices,
        reward=_as_obj(cfgd["reward"]),
        trade=_as_obj(cfgd["trade_manager"]),
        segment_len=segment_len,
    )
    return tm.run()


class _Obj:
    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            setattr(self, k, v)


def _as_obj(d: Dict[str, Any]) -> _Obj:
    # shallow object wrapper: cfg.reward.transaction_cost works
    return _Obj(d)


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--features", type=str, default="data/features.npy")
    ap.add_argument("--prices", type=str, default="data/prices.npy")

    ap.add_argument("--buy_model", type=str, required=True)
    ap.add_argument("--sell_model", type=str, required=True)

    ap.add_argument("--out_root", type=str, default="runs_ga")
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)

    # GA knobs
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=10)

    ap.add_argument("--meta", type=str, default="data/row_meta.parquet")
    ap.add_argument("--val_frac", type=float, default=0.15)

    # Fitness target (keep it simple)
    # ap.add_argument("--fitness", type=str, default="test_final_equity", choices=["test_final_equity"])

    args = ap.parse_args()

    cfgd = load_yaml(args.config)
    cfg = load_yaml_to_ns(args.config)

    run_id = args.run_id or now_id("ga")
    out_dir = os.path.join(args.out_root, run_id)
    ensure_dir(out_dir)

    # ---- data
    features = np.load(args.features).astype(np.float32, copy=False)
    prices = np.load(args.prices).astype(np.float32, copy=False)

    tickers = list(cfg.data.tickers)
    train_frac = float(get_by_path(cfgd, "data.train_frac", 0.7))

    # X_train, p_train, X_test, p_test, seg_train_len, seg_test_len, n_segs = split_by_ticker_time(
    #     features=features,
    #     prices=prices,
    #     tickers=tickers,
    #     meta_path=args.meta,
    #     train_frac=train_frac,
    # )
    
    # The specific plist for GA in order to have a VAL dataset
    X_train, p_train, X_val, p_val, X_test, p_test, seg_train_len, seg_val_len, seg_test_len, n_segs = split_by_ticker_time_ga(
        features=features,
        prices=prices,
        tickers=tickers,
        meta_path=args.meta,
        train_frac=train_frac,
        val_frac=float(args.val_frac),
    )


    print("=== DATA SPLIT ===")
    print("tickers in cfg:", len(tickers))
    
    print("rows_per_ticker:", seg_train_len + seg_val_len + seg_test_len)
    print("train/val/test per ticker:", seg_train_len, seg_val_len, seg_test_len)
    print("train rows total:", len(p_train), "val rows total:", len(p_val), "test rows total:", len(p_test))

    print("features:", features.shape, "prices:", prices.shape)
    print("N_SEGS:", n_segs, "TRAIN_FRAC:", train_frac)
    print("train_len per seg:", seg_train_len, "test_len per seg:", seg_test_len)
    print("X_train:", X_train.shape, "p_train:", p_train.shape)
    print("X_test :", X_test.shape, "p_test :", p_test.shape)

    # ---- load agents (fixed, GA changes only TM params)
    tmp_buy_env = BuyEnv(features=X_train, prices=p_train, reward=cfg.reward, trade=cfg.trade_manager)
    print("BuyEnv state_dim:", tmp_buy_env.state_dim, "X_train feature_dim:", X_train.shape[1])

    buy_cfg = deepcopy(cfgd["agent"])
    # buy_cfg["state_dim"] = int(X_train.shape[1])
    buy_cfg["state_dim"] = int(tmp_buy_env.state_dim)
    buy_cfg["n_actions"] = 2
    buy_agent = DDQNAgent(_as_obj(buy_cfg))
    buy_agent.load(args.buy_model)

    # infer sell state_dim: base + 3
    sell_cfg = deepcopy(cfgd["agent"])
    sell_cfg["state_dim"] = int(X_train.shape[1]) + 3
    sell_cfg["n_actions"] = 2
    sell_agent = DDQNAgent(_as_obj(sell_cfg))
    sell_agent.load(args.sell_model)

    # ---- define genes (ADD MORE HERE LATER)
    # genes = [
    #     GeneSpec(
    #         name="buy_min_confidence",
    #         path="trade_manager.buy_min_confidence",
    #         kind="float",
    #         bounds=(0.35, 0.75),
    #     ),
    # ]
    genes = [
        GeneSpec("buy_min_confidence", "trade_manager.buy_min_confidence", "float", bounds=(0.40, 0.75)),
        GeneSpec("sell_min_margin", "trade_manager.sell_min_margin", "float", bounds=(0.00, 0.08)),
        GeneSpec("sell_min_delta_vs_hold", "trade_manager.sell_min_delta_vs_hold", "float", bounds=(0.00, 0.05)),
        GeneSpec("min_hold_bars", "trade_manager.min_hold_bars", "int", bounds=(0, 10)),
        GeneSpec("cooldown_steps", "trade_manager.cooldown_steps", "int", bounds=(0, 10)),
    ]


    # ---- logging
    log_path = os.path.join(out_dir, "ga_log.jsonl")
    best_path = os.path.join(out_dir, "best.json")
    meta_path = os.path.join(out_dir, "meta.json")

    meta = {
        "run_id": run_id,
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "config": args.config,
        "features": args.features,
        "prices": args.prices,
        "buy_model": args.buy_model,
        "sell_model": args.sell_model,
        "ga": {"pop": int(args.pop), "gens": int(args.gens)},
        "genes": [{"name": g.name, "path": g.path, "kind": g.kind, "bounds": g.bounds, "choices": g.choices} for g in genes],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    def evaluator(genome: Genome):
        # apply overrides
        cfg_try = deepcopy(cfgd)
        overrides = genome.as_config_overrides()
        for path, v in overrides.items():
            set_by_path(cfg_try, path, v)

        
        # ----------------------------
        # Split VAL into two halves per ticker
        # ----------------------------
        # half = seg_val_len // 2

        # val1_idx = []
        # val2_idx = []

        # for i in range(n_segs):
        #     start = i * seg_val_len
        #     mid   = start + half
        #     end   = start + seg_val_len

        #     val1_idx.append(np.arange(start, mid))
        #     val2_idx.append(np.arange(mid, end))

        # val1_idx = np.concatenate(val1_idx)
        # val2_idx = np.concatenate(val2_idx)

        # X_val1, p_val1 = X_val[val1_idx], p_val[val1_idx]
        # X_val2, p_val2 = X_val[val2_idx], p_val[val2_idx]

        # seg_val1_len = half
        # seg_val2_len = seg_val_len - half

        # tm_v1 = run_tm_with_sell(cfg_try, X_val1, p_val1, buy_agent, sell_agent, seg_val1_len)
        # tm_v2 = run_tm_with_sell(cfg_try, X_val2, p_val2, buy_agent, sell_agent, seg_val2_len)

        # print("DBG v1 entry_debug:", tm_v1.get("entry_debug"))
        # print("DBG v1 trade_cfg:", cfg_try["trade_manager"])
        # print("DBG seg_val1_len:", seg_val1_len)

        # eq1 = float(tm_v1.get("final_equity", 0.0))
        # eq2 = float(tm_v2.get("final_equity", 0.0))

        # log1 = float(np.log(max(eq1, 1e-9)))
        # log2 = float(np.log(max(eq2, 1e-9)))

        # t1 = tm_v1.get("trades", [])
        # t2 = tm_v2.get("trades", [])
        # s1 = summarize_trades(t1)
        # s2 = summarize_trades(t2)

        # fitness = min(log1, log2) - 0.001 * (s1["n_trades"] + s2["n_trades"])

        # metrics = {
        #     "val1_final_equity": eq1,
        #     "val2_final_equity": eq2,
        #     "val1_log_equity": log1,
        #     "val2_log_equity": log2,
        #     "val1_n_trades": s1["n_trades"],
        #     "val2_n_trades": s2["n_trades"],
        #     "val1_avg_net": s1["avg_net_return"],
        #     "val2_avg_net": s2["avg_net_return"],
        #     "val1_win_rate": s1["win_rate"],
        #     "val2_win_rate": s2["win_rate"],
        # }
        tm_val = run_tm_with_sell(cfg_try, X_val, p_val, buy_agent, sell_agent, segment_len=seg_val_len)
        eq = float(tm_val.get("final_equity", 1.0))
        log_eq = float(np.log(max(eq, 1e-9)))
        s = summarize_trades(tm_val.get("trades", []))

        fitness = log_eq - 0.001 * s["n_trades"]
        metrics = {
            "val_final_equity": eq,
            "val_log_equity": log_eq,
            "val_n_trades": s["n_trades"],
            "val_avg_net": s["avg_net_return"],
            "val_win_rate": s["win_rate"],
            "entry_debug": tm_val.get("entry_debug", {}),
            "exit_reasons": tm_val.get("exit_reasons", {}),
        }

        # log one line
        rec = {
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "genes": dict(genome.values),
            "fitness": fitness,
            "metrics": metrics,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        return fitness, metrics

    ga_cfg = GAConfig(
        population_size=int(args.pop),
        generations=int(args.gens),
        seed=int(args.seed),
        elite_frac=0.25,
        tournament_k=3,
        crossover_rate=0.6,
        mutation_rate=0.3,
        mutation_sigma=0.05,
    )

    pop = Population(genes=genes, cfg=ga_cfg, evaluator=evaluator)
    pop.init()

    # ---- run GA
    # for gen in range(int(ga_cfg.generations)):
    #     best = pop.best()
    #     print(f"\n=== GEN {gen} ===")
    #     print("BEST:", best.genome.values, "fitness:", best.fitness)
    #     print("TEST:", best.fitness,
    #           "n_trades:", best.metrics.get("test_n_trades"),
    #           "avg_net:", best.metrics.get("test_avg_net"),
    #           "win_rate:", best.metrics.get("test_win_rate"))

    #     pop.evolve_one_generation()
    for gen in range(int(ga_cfg.generations)):
        best = pop.best()
        print(f"\n=== GEN {gen} ===")
        print("BEST:", best.genome.values, "fitness:", best.fitness)
        print("VAL:",
            best.fitness,
            "eq1:", best.metrics.get("val1_final_equity"),
            "eq2:", best.metrics.get("val2_final_equity"),
            "trades:", best.metrics.get("val1_n_trades"), "/", best.metrics.get("val2_n_trades"),
            "min_log:", min(best.metrics.get("val1_log_equity", -999), best.metrics.get("val2_log_equity", -999)))


        pop.evolve_one_generation()

    # best = pop.best()
    # out = {
    #     "best_genes": dict(best.genome.values),
    #     "fitness": float(best.fitness),
    #     "metrics": dict(best.metrics),
    # }
    best = pop.best()

    # ----------------------------
    # FINAL EVAL ON TEST (ONCE)
    # ----------------------------
    cfg_best = deepcopy(cfgd)
    for path, v in best.genome.as_config_overrides().items():
        set_by_path(cfg_best, path, v)

    tm_test = run_tm_with_sell(
        cfg_best,
        X_test, p_test,
        buy_agent, sell_agent,
        seg_test_len
    )

    test_final_equity = float(tm_test.get("final_equity", 0.0))
    test_trades = tm_test.get("trades", [])
    test_stats = summarize_trades(test_trades)

    print("Final TEST equity:", test_final_equity,
      "n_trades:", test_stats["n_trades"],
      "avg_net:", test_stats["avg_net_return"],
      "win_rate:", test_stats["win_rate"])


    test_metrics = {
        "test_final_equity": test_final_equity,
        "test_n_trades": test_stats["n_trades"],
        "test_avg_net": test_stats["avg_net_return"],
        "test_win_rate": test_stats["win_rate"],
        "test_min_net": test_stats["min_net"],
        "test_median_net": test_stats["median_net"],
        "test_max_net": test_stats["max_net"],
        "exit_reasons": tm_test.get("exit_reasons", {}),
        "entry_debug": tm_test.get("entry_debug", {}),
        "sell_debug": tm_test.get("sell_debug", {}),
    }

    out = {
        "best_genes": dict(best.genome.values),

        # GA objective value (VAL fitness)
        "fitness": float(best.fitness),
        "val_metrics": dict(best.metrics),

        # Final untouched evaluation
        "test_metrics": test_metrics,
    }

    with open(best_path, "w") as f:
        json.dump(out, f, indent=2)

    # print("\n=== GA DONE ===")
    # print("Best genes:", out["best_genes"])
    # print("Best fitness (test final_equity):", out["fitness"])
    # print("Saved:", best_path)
    # print("Log:", log_path)
    # print("Meta:", meta_path)
    print("\n=== GA DONE ===")
    print("Best genes:", out["best_genes"])
    print("Best VAL fitness:", out["fitness"])
    print("Final TEST equity:", out["test_metrics"]["test_final_equity"])
    print("Saved:", best_path)
    print("Log:", log_path)
    print("Meta:", meta_path)



if __name__ == "__main__":
    main()
