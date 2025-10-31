## Slide 2: The project

## Slide 3: Template and Idea
The idea comes from the ‘Financial Advisor Bot’ template, which focuses on active portfolio management. I’ll extend it with adaptive learning and explainable AI so that it can not only trade but also communicate why it makes certain decisions.

## Slide4: Motivation
The financial world is unpredictable and noisy — the perfect challenge for Reinforcement Learning. I’ve always been fascinated by the idea that intelligent systems can uncover hidden patterns in what seems to be chaos.


## Slide 5: Background Research
TBD

## Slide 6: References
I reviewed key works that shaped financial AI — from Moody & Saffell’s early reinforcement trading models to Jiang’s deep architectures and FinRL’s open frameworks. I also looked at Jin et al.’s GA-optimised DQN, which inspired my idea to merge RL and GA. Finally, StockBench reminded me that even the latest LLMs still struggle with market complexity — showing why adaptive learning remains essential.

## Slide 7: Challenges
The biggest challenges are instability and realism. Financial data is chaotic, and most AI models overfit or ignore real-world constraints. Another challenge is trust — users won’t rely on a black box that can’t explain its reasoning.

## Slide 8: Proposed solution
My approach is to merge these three dimensions: Reinforcement Learning to learn trading strategies, Genetic Algorithms to evolve and stabilise them, and Natural-Language Processing to translate complex outputs into readable advice for human users.

## Slide 9:
Training will follow a two-level process. Inside each agent, reinforcement learning optimises a trading policy step-by-step. Across agents, a genetic algorithm evolves the best hyperparameter configurations. Each generation evaluates performance through a fitness function based on risk-adjusted returns — mainly the Sharpe ratio, drawdown, and transaction cost. The result is a population that converges towards more stable and profitable trading behaviours.

## Slide 10: methodology
Given the ambition and exploratory nature of this project, I’m going to follow an Agile approach rather than a traditional waterfall model. In practice, that means short, iterative development cycles — design, test, refine, and adapt — rather than trying to define everything from the beginning.
Financial markets are unpredictable, and some factors — like which indicators or features have the biggest impact — can only be discovered experimentally. Reinforcement Learning and Genetic Algorithms fit this mindset perfectly because both involve continuous adaptation and optimisation over time.
So, the system will evolve in parallel with the project itself: each iteration will refine the models, the reward design, and even the data preprocessing strategy.

## Slide 11: Testing
TBC: I’ll evaluate the GA-optimised RL agent against both a pure RL baseline and standard trading strategies. The fitness function already enforces risk-adjusted profitability, but I’ll also test across different market periods to assess robustness. Finally, the NLP layer will be evaluated qualitatively to ensure it translates model output into understandable explanations.

## Sldie 12: Validation
TBC: Testing will be based on backtesting across multiple time periods and assets, evaluating both profitability and stability. I’ll also assess the clarity of the NLP explanations, to ensure the bot communicates its advice effectively.

## Slide 13: Conclusion
This project is my way to combine everything I’ve learned — from reinforcement and evolutionary learning to NLP — into one system that can learn, adapt, and communicate intelligently. Thank you.