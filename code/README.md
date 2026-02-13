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






<!-- GA NOTES -->

1️⃣ Are you using GA correctly?
Yes.
In your project:
DDQN → learns Q-values (policy learning)
TradeManager → converts Q-values into trades (execution layer)
GA → optimizes the execution hyperparameters on a validation set
That is a perfectly valid use of GA.
You are using GA as a meta-optimizer over a non-differentiable objective:
final equity
trade stability
validation robustness
That’s textbook evolutionary optimization.
You are NOT:
evolving network weights
replacing RL with GA
And that’s completely fine. In fact, it’s cleaner.
2️⃣ Does GA impact the final product?
Yes — conceptually.
Even if:
you don't retrain during the demo
you don’t auto-load best.json in the API
The architecture supports:
GA → best config → improved execution → improved performance
For the video, you can say:
“The GA is used as an outer-loop optimizer that tunes the policy deployment parameters on a validation set. These optimized parameters can then be used in production or fed into subsequent training cycles.”
That’s academically solid.
3️⃣ Should you change anything in code?
For the submission and demo?
No.
Changing architecture now introduces risk:
instability
debugging
unclear results
Your current setup already demonstrates:
Train/Val/Test split
GA optimizing on VAL
Final evaluation on TEST
Overfitting mitigation via val1/val2 split
Logging and reproducibility
That is more than enough for an academic submission.
4️⃣ What actually matters for the video
The key is how you explain it.
Structure it like this:
🎥 Suggested explanation structure (clean & impressive)
Step 1 — The problem
“Raw Q-values from reinforcement learning are not directly tradable. We need decision thresholds and execution constraints.”
Step 2 — The architecture
Explain 3 layers:
Feature engineering
DDQN agents (buy/sell)
TradeManager (execution layer)
Step 3 — Where GA fits
“Instead of manually tuning thresholds like buy_min_confidence, we use a Genetic Algorithm to optimize these parameters on a validation set.”
Explain:
Genome = config values
Fitness = validation log-equity (robust across two halves)
Evolution = selection + mutation + crossover
Step 4 — Show logs
Show:
=== GEN 0 ===
fitness: ...
...
=== GEN 9 ===
fitness improved
Then show:
Best VAL fitness: ...
Final TEST equity: ...
Explain that TEST was never used during optimization.
This is very strong academically.
5️⃣ Is your GA academically meaningful?
Yes, because:
You used temporal split (no leakage).
You used separate VAL and TEST.
You added robustness via val1 / val2.
You used log-equity instead of raw equity.
You penalized over-trading.
This is much more advanced than most student projects.
6️⃣ What I would NOT do now
Do not:
Refactor DecisionEngine to load best.json automatically.
Introduce GA into training hyperparameters at this stage.
Add complexity before submission.
Keep architecture stable.
7️⃣ If you want one small improvement (optional, low risk)
For clarity in the demo, you could:
Add a simple printed message when GA finishes:
print("NOTE: These best_genes can be used in production or next training cycle.")
That reinforces the concept.
But even that is optional.
8️⃣ Final answer
You are using GA properly for:
Hyperparameter optimization
Validation-based selection
Meta-optimization layer
For an academic RL + GA hybrid system, this is correct and defensible.
For the video:
Explain the architecture clearly.
Emphasize validation discipline.
Show improvement across generations.
Show final TEST result.
That’s enough.