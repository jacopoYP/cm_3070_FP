# Clean Trading RL (Refactor Baseline)

This is a **single-process**, **minimal** refactor of your trading RL project.

It keeps:
- DDQN + Replay Buffer
- Buy agent + Sell agent (sell env uses buy entry indices)
- TradeManager orchestration (buy triggers entry, sell triggers exit)
- Config-driven parameters (GA-ready)
- Simple softmax confidence gating for BUY

It removes:
- multiprocessing / workers / shared buffers (until baseline is stable)

## Quick start (buy baseline)

You need two numpy arrays:
- `features.npy`: shape `(n_steps, state_dim)`
- `prices.npy`: shape `(n_steps,)`

Run:

```bash
python -m clean_trading_rl.scripts.run_train_buy \
  --config clean_trading_rl/config.yaml \
  --features_npy /path/to/features.npy \
  --prices_npy /path/to/prices.npy \
  --out_dir runs
```

Outputs (in `runs/<timestamp>/`):
- `buy_agent.pt`
- `q_gap_buy.png` and `q_gap_buy_hist.png`

## What to lock before adding complexity
1. Q-gap plot moves away from 0 (no more confidence stuck at ~0.50)
2. Non-zero trades in backtest
3. Reproducible runs with a fixed seed

Once the baseline is stable, we can add:
- sell training script
- sentiment index integration
- GA hyperparameter search
