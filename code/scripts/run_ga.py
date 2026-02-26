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
import numpy as np
# import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ga.genome import GeneSpec, Genome
from ga.population import GAConfig, Population

from agents.ddqn_agent import DDQNAgent
from trade.trade_manager import TradeManager

from core.helper import split_by_ticker_time_ga, load_yaml, load_yaml_to_ns, now_run_id, check_dir
from core.metrics import summarize_trades
from envs.buy_env import BuyEnv

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

# ---------------------------------------------------------------------
# evaluation (Trade Manager WITH SELL)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
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

    args = ap.parse_args()

    cfgd = load_yaml(args.config)
    cfg = load_yaml_to_ns(args.config)

    run_id = args.run_id or now_run_id("ga")
    out_dir = os.path.join(args.out_root, run_id)
    check_dir(out_dir)

    # ---- data
    features = np.load(args.features).astype(np.float32, copy=False)
    prices = np.load(args.prices).astype(np.float32, copy=False)

    tickers = list(cfg.data.tickers)
    train_frac = float(get_by_path(cfgd, "data.train_frac", 0.7))
    
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

    genes = [
        GeneSpec("buy_min_confidence", "trade_manager.buy_min_confidence", "float", bounds=(0.40, 0.75)),
        GeneSpec("sell_min_margin", "trade_manager.sell_min_margin", "float", bounds=(0.00, 0.08)),
        GeneSpec("sell_min_delta_vs_hold", "trade_manager.sell_min_delta_vs_hold", "float", bounds=(0.00, 0.05)),
        GeneSpec("min_hold_bars", "trade_manager.min_hold_bars", "int", bounds=(0, 10)),
        GeneSpec("cooldown_steps", "trade_manager.cooldown_steps", "int", bounds=(0, 10)),
    ]

    # Logging
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

    def fitness_function(genome: Genome):
        cfg_try = deepcopy(cfgd)
        overrides = genome.as_config_overrides()
        for path, v in overrides.items():
            set_by_path(cfg_try, path, v)

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

    pop = Population(genes=genes, cfg=ga_cfg, evaluator=fitness_function)
    pop.init()

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

    print("\n=== GA DONE ===")
    print("Best genes:", out["best_genes"])
    print("Best VAL fitness:", out["fitness"])
    print("Final TEST equity:", out["test_metrics"]["test_final_equity"])
    print("Saved:", best_path)
    print("Log:", log_path)
    print("Meta:", meta_path)


if __name__ == "__main__":
    main()
