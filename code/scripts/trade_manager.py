import yaml
import numpy as np

from agents.buy_agent_trainer import BuyAgentTrainer
from agents.sell_agent_trainer import SellAgentTrainer


class TradeManager:
    """
    TradeManager coordinates the trained Buy and Sell agents over a single
    historical pass, producing a backtest with an equity curve and trade log.

    - When flat:
        BuyAgent may open a new long position (if not in cooldown).
    - When long:
        SellAgent decides each step whether to close the trade.
    - After closing:
        a cooldown period (from config) prevents immediate re-entry.
    """

    def __init__(
        self,
        buy_trainer: BuyAgentTrainer,
        sell_trainer: SellAgentTrainer,
        cfg_path: str = "config/data_config.yaml",
    ):
        # Trainers (with trained agents, state_df, prices)
        self.buy_trainer = buy_trainer
        self.sell_trainer = sell_trainer

        assert buy_trainer.state_df is not None
        assert buy_trainer.prices is not None
        assert buy_trainer.agent is not None
        assert sell_trainer.agent is not None

        # Use BuyTrainer's dataset as the reference (shared index)
        self.state_df = buy_trainer.state_df
        self.prices = buy_trainer.prices

        self.buy_agent = buy_trainer.agent
        self.sell_agent = sell_trainer.agent

        # Load config
        self.cfg = yaml.safe_load(open(cfg_path))
        tm_cfg = self.cfg.get("trade_manager", {})

        # Configurable cooldown + sell horizon + transaction cost
        self.cooldown_steps: int = tm_cfg.get("cooldown_steps", 5)
        self.sell_horizon: int = tm_cfg.get(
            "sell_horizon",
            getattr(sell_trainer, "horizon", 20),
        )
        # Cost used for evaluation (buy + sell combined)
        self.transaction_cost: float = tm_cfg.get("transaction_cost", 0.001)

    # ------------------------------------------------------------------ #
    # Main backtest
    # ------------------------------------------------------------------ #

    def run_backtest(self, greedy: bool = True):
        """
        Run a single-pass backtest over the dataset using Buy + Sell agents.

        Returns:
            {
              "equity_curve": np.array of shape (T+1,),
              "trades": list of dicts with entry/exit info
            }
        """

        n_steps = len(self.state_df)
        position = 0          # 0 = flat, 1 = long
        entry_idx = None
        entry_price = None
        cooldown = 0

        equity_curve = [1.0]
        last_equity = 1.0
        trades = []

        for t in range(n_steps):
            state = self.state_df.iloc[t].values.astype(np.float32)
            price_t = float(self.prices.iloc[t])

            # Default: equity carries over from previous step
            equity_changed_this_step = False

            if position == 0:
                # We are FLAT
                if cooldown > 0:
                    cooldown -= 1
                    # No trading, just carry equity
                else:
                    # BuyAgent may open a new trade
                    if greedy:
                        action = self.buy_agent.select_action(state, greedy=True)
                    else:
                        action = self.buy_agent.select_action(state)

                    # In BuyEnv, 1 = BUY
                    if action == 1:
                        position = 1
                        entry_idx = t
                        entry_price = price_t

            else:
                # We are LONG: SellAgent controls the exit
                hold_bars = t - entry_idx if entry_idx is not None else 0

                if greedy:
                    sell_action = self.sell_agent.select_action(state, greedy=True)
                else:
                    sell_action = self.sell_agent.select_action(state)

                force_close = (
                    hold_bars >= self.sell_horizon or t == n_steps - 1
                )

                if sell_action == 1 or force_close:
                    # Close trade
                    exit_idx = t
                    exit_price = price_t

                    if entry_price is not None and entry_price != 0.0:
                        gross_return = (exit_price - entry_price) / entry_price
                    else:
                        gross_return = 0.0

                    # Simple net return with buy+sell costs
                    net_return = gross_return - 2.0 * self.transaction_cost

                    trades.append(
                        {
                            "entry_idx": entry_idx,
                            "exit_idx": exit_idx,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "gross_return": gross_return,
                            "net_return": net_return,
                            "hold_bars": hold_bars,
                            "forced_exit": force_close,
                        }
                    )

                    last_equity = last_equity * (1.0 + net_return)
                    equity_changed_this_step = True

                    # Reset position, start cooldown
                    position = 0
                    entry_idx = None
                    entry_price = None
                    cooldown = self.cooldown_steps

            # Append equity for this step
            if not equity_changed_this_step:
                # No trade closed → carry forward
                equity_curve.append(last_equity)
            else:
                equity_curve.append(last_equity)

        results = {
            "equity_curve": np.array(equity_curve),
            "trades": trades,
        }
        return results
