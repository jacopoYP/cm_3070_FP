# cm_3070_FP
Final project

Tutor e-mail: haris.binzia@london.ac.uk

How to reference: https://onlinelibrary.london.ac.uk/support/referencing 

## Removed from Design Intro so far

I started designing this project asking my-self the following questions:
How reinforcement-learning agents can learn profitable and stable investment strategies in realistic, non-stationary financial environments?

Can Genetic Algorithms improve the performance and stability of RL-based trading agents by evolving their hyperparameters 
or reward functions automatically? 
Natural Language Processing or LLM techniques need to act as ‘bridge’ to translate quantitative agent outputs into clear, human-readable recommendations / instructions 
What system architecture best combines RL, GA, and NLP to support model performance and transparency at the same time? 
How can the success of an AI-driven financial advisor be measured not only in profitability but also in interpretability and user trust? Based on several of the studies I have reviewed, I have learned that back-testing remains one of the biggest challenges: not only because of the computational power required, but also because of the need for realistic assumptions about transaction costs, liquidity, and changing market regimes.

#Development

✅ 1. Indicators: should we adjust or experiment later?
Yes — this is normal and expected.
Right now your set:
return_1d, rsi14, macd_diff, bb_b, atr14, roc10, obv, mfi14, willr14
is perfect for a first prototype because:
They are widely used in literature (Sezer, Ozbayoglu, FINRL baseline)
They cover momentum, volatility, volume, and trend
They are computationally cheap
They avoid overfitting (unlike using dozens of indicators)
Later, once your RL agents learn something, you can experiment with:
Fama–French factors (value, size, market beta)
More momentum windows
Volatility clustering features (GARCH-like)
But NOT now.
✔ Keep the indicator set stable until the RL loop works.
Otherwise you increase the moving parts and debugging becomes impossible.


Conclusion: What you will implement
Stage				Use Sharpe?		Why
DDQN reward		❌ No				too unstable
GA fitness		✔ Yes				ideal for evolutionary tuning
Post-backtest 	✔ Yes				needed to compare baselines
evaluation

So your RL agent learns on price-based reward, and Sharpe/Sortino enter later as training “quality signals”.

But for the student project, Option A is more than good enough.
Use SB3 VecEnv for the first prototype.


Final Summary
✅ Sharpe/Sortino usage
Not used as DDQN reward
Used in GA fitness
Used in evaluation and comparison
✅ Transaction costs / slippage / liquidity
Included inside env.step()
costs = 0.1–0.2%
slippage = 0.05%
position limit to avoid unrealistic trades
✅ Multi-process
Use SB3 VecEnv for now
→ parallel BuyEnv / SellEnv instances
→ fast, clean, stable
Implement custom multiprocessing later IF needed.


The Parallel Multi-Module Reinforcement Learning Algorithm for Stock Trading (2024) introduced:
Component	Description	Where you use it
Multiple agents	Buy, Sell, Stop-loss	You have Buy + Sell + Sentiment
Parallelised RL environments	Each module is trained across parallel datasets/time windows	You use VecEnv / multiprocessing
DDQN	To avoid Q-value overestimation	You also use DDQN
CAPM-based feature expansion	Market + fundamental + correlated assets	You optionally add technical + sentiment data
So your design does follow the same spirit:
multiple specialised modules
parallel RL environments
DDQN stabilised learning
market-aware state representation
Theirs is a single-stock agent; your system is multi-agent but not portfolio-level yet.
That’s fine — your system still fits the conceptual framework.
⚠️ Important: Differences to mention in your report
You should highlight (because this is academically correct AND makes your design stronger):
(1) Their parallelism is hand-coded; yours uses vectorized environments
The 2024 paper launches N independent RL training “threads” and aggregates transitions manually.
Your version uses:
✔ SB3 VecEnv or
✔ Python multiprocessing
Which is:
more stable
easier to implement
industry-standard (same as FinRL, RLlib, Ray)
still conceptually identical
This is perfect for a university project.

The Parallel Multi-Module Reinforcement Learning Algorithm for Stock Trading (2024) introduced:
Component	Description	Where you use it
Multiple agents	Buy, Sell, Stop-loss	You have Buy + Sell + Sentiment
Parallelised RL environments	Each module is trained across parallel datasets/time windows	You use VecEnv / multiprocessing
DDQN	To avoid Q-value overestimation	You also use DDQN
CAPM-based feature expansion	Market + fundamental + correlated assets	You optionally add technical + sentiment data
So your design does follow the same spirit:
multiple specialised modules
parallel RL environments
DDQN stabilised learning
market-aware state representation
Theirs is a single-stock agent; your system is multi-agent but not portfolio-level yet.
That’s fine — your system still fits the conceptual framework.
⚠️ Important: Differences to mention in your report
You should highlight (because this is academically correct AND makes your design stronger):
(1) Their parallelism is hand-coded; yours uses vectorized environments
The 2024 paper launches N independent RL training “threads” and aggregates transitions manually.
Your version uses:
✔ SB3 VecEnv or
✔ Python multiprocessing
Which is:
more stable
easier to implement
industry-standard (same as FinRL, RLlib, Ray)
still conceptually identical
This is perfect for a university project.

This is acceptable and defensible in the design section:
“Due to constraints in time and data availability, I adopt a simplified but modular data architecture where additional fundamental factors (e.g., CAPM, Fama–French 5-factor) could be added in future iterations.”
(3) You add something they do NOT have:
✔ Genetic Algorithm hyperparameter evolution
✔ NLP explainability layer
✔ Sentiment validation module
✔ Multi-agent structure (Buy + Sell + Sentiment)
These are meaningful contributions.

Maybe change design section in this way

What to write in your design section:
“This project adopts the parallel multi-process philosophy proposed in the 2024 Parallel Multi-Module RL study. Their research shows that running multiple environments in parallel improves exploration, stability, and convergence speed in financial RL tasks.
Instead of implementing custom multiprocessing as in their work, the project uses Stable-Baselines3’s VecEnv architecture, which offers a battle-tested and reproducible form of parallel RL. Conceptually, both approaches serve the same purpose: each agent learns from diverse trajectories that reduce overfitting to a single market regime.”
“Additionally, the project draws inspiration from their CAPM extension, which argues that returns are influenced not only by price history but also by market factors and related instruments. While full CAPM integration (e.g., Fama–French factors) is outside the scope of the first prototype, the design remains modular to incorporate these enhancements in future iterations.”

pip install yfinance
pip install PyYAML
pip install ta
pip install vaderSentiment

pip install "numpy<2.0"



# PROD 
✅ 1. Production Warmup Strategy
Prototype
✔ warmup_steps = 10
(because mean transitions/episode ≈ 20 → warmup 500 never triggers)
Production
You should compute the warmup dynamically:
warmup_steps = int(0.1 * len(state_df))   # 10% of dataset
or even better:
warmup_steps = max(200, int(len(state_df) * 0.2))
Why?
In production we will use bigger datasets (multiple tickers, longer windows)
Episodes will last longer once we remove “done after BUY”
Replay buffer must reach a representative distribution before training
👉 Recommended production warmup: 200–1000 steps

✅ 2. Production BuyEnv Must NOT End on BUY
Right now:
if action == 1:
    reward = ...
    done = True
This creates a trivial ultra-short episode → bad for RL.
Production behavior:
Keep track of holding logic (“position” variable)
Allow multiple BUY/HOLD/SELL over a single episode
Episode ends only when dataset ends or after fixed length
This produces hundreds or thousands of transitions per episode → ideal for DDQN.

✅ 3. Production Reward Must Be Risk-Adjusted
Prototype:
reward = (exit_price - entry_price) / entry_price - cost
Production version should include:
Sharpe or Sortino component
Drawdown penalty
Position duration penalty
Slippage modeling
Example:
reward = (
    (exit_price - entry_price) / entry_price
    - cost
    - 0.001 * max_drawdown
    - 0.0001 * volatility
)

✅ 4. Production Must Use Multi-Process Training
Prototype = single-thread → OK
Production must use multi-process episodes (“Parallel Multi-Module RL”, 2024 paper):
Use Ray RLlib or Python multiprocessing
Train multiple copies of BuyEnv simultaneously
Combine gradients (A3C style) or replay buffer sharing (DQN style)
Why?
Dramatically faster convergence
Better robustness across market regimes

✅ 5. Production Must Have Larger Replay Buffer
Prototype:
replay_buffer around 20,000
Production:
replay_buffer = 200,000–500,000
Because multi-step trading environments generate high correlation → larger buffer fights overfitting.

✅ 6. GA Hyperparameter Search
Prototype = no GA
Production = GA evolves:
learning rate
gamma
epsilon decay
reward weights
target update frequency
GA framework runs periodically:
train RL → evaluate fitness → mutate → next generation

✅ 7. Production Sentiment Ingestion
Prototype:
Sentiment indexing disabled
Production:
Lightweight polarity for preprocessing (FinBERT/VADER)
High-level sentiment validator as separate agent
⚡ Summary Table
Component	Prototype	Production
Warmup steps	10	200–1000 or dynamic % of dataset
Episode length	Ends on BUY	Full multi-step trading episode
Reward	Simple next-K return	Risk-adjusted + penalties
Replay buffer	~20k	200k–500k
Parallelism	None	Multi-process (Ray/multiprocessing)
GA	Disabled	Enabled for hyperparams
Sentiment	Not used	Enabled (index + deep sentiment)
Debugging	Simple logs	TensorBoard, wandb, metrics


First log BuyAgent

[BuyTrainer] Raw dataset: (1224, 10) [BuyTrainer] 
After dropna: (1224, 10) [BuyTrainer] 
 state_df shape: (1194, 270) [BuyTrainer] state_dim=270, actions=2 [BuyTrainer] 
 Dynamic warmup set to: 238 
 [Episode 1/50] Reward=2.6845 | MeanTrades=2.6845 | Eps=0.887 | Steps=1193 | Trades=1 | Buffer=1193 | Avg10=2.6845 
 [Episode 5/50] Reward=2.6845 | MeanTrades=2.6845 | Eps=0.433 | Steps=1193 | Trades=1 | Buffer=5965 | Avg10=-1.2854 
 [Episode 10/50] Reward=nan | MeanTrades=nan | Eps=0.050 | Steps=1193 | Trades=115 | Buffer=11930 | Avg10=nan 
 [Episode 15/50] Reward=-1.3378 | MeanTrades=-0.0515 | Eps=0.050 | Steps=1193 | Trades=26 | Buffer=17895 | Avg10=nan 
 [Episode 20/50] Reward=-0.6549 | MeanTrades=-0.0252 | Eps=0.050 | Steps=1193 | Trades=26 | Buffer=23860 | Avg10=-0.8987 
 [Episode 25/50] Reward=-0.5816 | MeanTrades=-0.0264 | Eps=0.050 | Steps=1193 | Trades=22 | Buffer=29825 | Avg10=-0.7663 
 [Episode 30/50] Reward=-1.2247 | MeanTrades=-0.0395 | Eps=0.050 | Steps=1193 | Trades=31 | Buffer=35790 | Avg10=-0.7585 
 [Episode 35/50] Reward=-0.4532 | MeanTrades=-0.0197 | Eps=0.050 | Steps=1193 | Trades=23 | Buffer=41755 | Avg10=-0.7397 
 [Episode 40/50] Reward=-0.3481 | MeanTrades=-0.0174 | Eps=0.050 | Steps=1193 | Trades=20 | Buffer=47720 | Avg10=-0.7255 
 [Episode 45/50] Reward=-1.0220 | MeanTrades=-0.0444 | Eps=0.050 | Steps=1193 | Trades=23 | Buffer=53685 | Avg10=-0.7666 
 [Episode 50/50] Reward=-0.3397 | MeanTrades=-0.0170 | Eps=0.050 | Steps=1193 | Trades=20 | Buffer=59650 | Avg10=-0.7192 
 Training COMPLETE. Final 5 episode rewards: [-1.1343781697063073, -0.44265279380824596, -1.2665205086092135, -0.46967228080419754, -0.33969440119874644] episode_rewards epsilon steps buffer_size num_trades \ 0 2.684456 0.886665 1193 1193 1 1 -17.164656 0.773330 1193 2386 167 2 2.684456 0.659995 1193 3579 1 3 2.684456 0.546660 1193 4772 1 4 2.684456 0.433325 1193 5965 1 ... 1 -0.102782 2 2.684456 3 2.684456 4 2.684456