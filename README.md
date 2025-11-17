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
