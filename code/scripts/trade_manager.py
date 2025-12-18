# import yaml
# import numpy as np
# import torch

# from agents.buy_agent_trainer import BuyAgentTrainer
# from agents.sell_agent_trainer import SellAgentTrainer

# class TradeManager:
#     """
#     TradeManager coordinates the trained Buy and Sell agents over a single
#     historical pass, producing a backtest with an equity curve and trade log.

#     - When flat:
#         BuyAgent may open a new long position (if not in cooldown).
#     - When long:
#         SellAgent decides each step whether to close the trade.
#     - After closing:
#         a cooldown period (from config) prevents immediate re-entry.
#     """

#     def __init__(
#         self,
#         buy_trainer: BuyAgentTrainer,
#         sell_trainer: SellAgentTrainer,
#         cfg_path: str = "config/data_config.yaml",
#     ):
#         # Trainers (with trained agents, state_df, prices)
#         self.buy_trainer = buy_trainer
#         self.sell_trainer = sell_trainer

#         assert buy_trainer.state_df is not None
#         assert buy_trainer.prices is not None
#         assert buy_trainer.agent is not None
#         assert sell_trainer.agent is not None

#         # Use BuyTrainer's dataset as the reference (shared index)
#         self.state_df = buy_trainer.state_df
#         self.prices = buy_trainer.prices

#         self.buy_agent = buy_trainer.agent
#         self.sell_agent = sell_trainer.agent

#         # Load config
#         self.cfg = yaml.safe_load(open(cfg_path))
#         tm_cfg = self.cfg.get("trade_manager", {})

#         self.buy_min_confidence: float = tm_cfg.get("buy_min_confidence", 0.55)
#         self.use_trend_filter: bool = tm_cfg.get("use_trend_filter", True)
#         self.ma_short: int = tm_cfg.get("ma_short", 10)
#         self.ma_long: int = tm_cfg.get("ma_long", 30)

#         # Precompute simple moving averages for trend filter
#         prices_np = self.prices.values.astype(np.float32)
#         self.sma_short = self._rolling_mean(prices_np, self.ma_short)
#         self.sma_long = self._rolling_mean(prices_np, self.ma_long)

#         # Configurable cooldown + sell horizon + transaction cost
#         self.cooldown_steps: int = tm_cfg.get("cooldown_steps", 5)
#         self.sell_horizon: int = tm_cfg.get(
#             "sell_horizon",
#             getattr(sell_trainer, "horizon", 20),
#         )
#         # Cost used for evaluation (buy + sell combined)
#         self.transaction_cost: float = tm_cfg.get("transaction_cost", 0.001)

#     # ------------------------------------------------------------------ #
#     # Main backtest
#     # ------------------------------------------------------------------ #

#     def run_backtest(self, greedy: bool = True):
#         """
#         Run a single-pass backtest over the dataset using Buy + Sell agents.

#         Returns:
#             {
#               "equity_curve": np.array of shape (T+1,),
#               "trades": list of dicts with entry/exit info
#             }
#         """

#         n_steps = len(self.state_df)
#         position = 0          # 0 = flat, 1 = long
#         entry_idx = None
#         entry_price = None
#         cooldown = 0

#         equity_curve = [1.0]
#         last_equity = 1.0
#         trades = []

#         for t in range(n_steps):
#             state = self.state_df.iloc[t].values.astype(np.float32)
#             price_t = float(self.prices.iloc[t])

#             # Default: equity carries over from previous step
#             equity_changed_this_step = False

#             if position == 0:
#                 # We are FLAT
#                 if cooldown > 0:
#                     cooldown -= 1

#                 else:
#                     # -----------------------------
#                     # 1) Optional trend filter
#                     # -----------------------------
#                     if self.use_trend_filter:
#                         sma_s = self.sma_short[t]
#                         sma_l = self.sma_long[t]
#                         # If we don't have enough history, or trend is not positive -> skip
#                         if np.isnan(sma_s) or np.isnan(sma_l) or sma_s <= sma_l:
#                             # skip any buy at this step
#                             pass
#                         else:
#                             # 2) Confidence-based BUY decision
#                             with torch.no_grad():
#                                 s = torch.from_numpy(state).to(self.buy_agent.device).unsqueeze(0)
#                                 q_vals = self.buy_agent.q_net(s)[0]   # shape [n_actions]
#                                 q_np = q_vals.detach().cpu().numpy()

#                             # Softmax to get "confidence-like" probability
#                             exps = np.exp(q_np - np.max(q_np))
#                             probs = exps / np.sum(exps)
#                             buy_prob = probs[1]    # action 1 = BUY
#                             # Optional: margin check between BUY and HOLD
#                             # margin = q_np[1] - q_np[0]

#                             if buy_prob >= self.buy_min_confidence:
#                                 position = 1
#                                 entry_idx = t
#                                 entry_price = price_t
#                     else:
#                         # No trend filter: only check confidence
#                         with torch.no_grad():
#                             s = torch.from_numpy(state).to(self.buy_agent.device).unsqueeze(0)
#                             q_vals = self.buy_agent.q_net(s)[0]
#                             q_np = q_vals.detach().cpu().numpy()

#                         exps = np.exp(q_np - np.max(q_np))
#                         probs = exps / np.sum(exps)
#                         buy_prob = probs[1]

#                         if buy_prob >= self.buy_min_confidence:
#                             position = 1
#                             entry_idx = t
#                             entry_price = price_t

#             else:
#                 # We are LONG: SellAgent controls the exit
#                 hold_bars = t - entry_idx if entry_idx is not None else 0

#                 if greedy:
#                     sell_action = self.sell_agent.select_action(state, greedy=True)
#                 else:
#                     sell_action = self.sell_agent.select_action(state)

#                 force_close = (
#                     hold_bars >= self.sell_horizon or t == n_steps - 1
#                 )

#                 if sell_action == 1 or force_close:
#                     # Close trade
#                     exit_idx = t
#                     exit_price = price_t

#                     if entry_price is not None and entry_price != 0.0:
#                         gross_return = (exit_price - entry_price) / entry_price
#                     else:
#                         gross_return = 0.0

#                     # Simple net return with buy+sell costs
#                     net_return = gross_return - 2.0 * self.transaction_cost

#                     trades.append(
#                         {
#                             "entry_idx": entry_idx,
#                             "exit_idx": exit_idx,
#                             "entry_price": entry_price,
#                             "exit_price": exit_price,
#                             "gross_return": gross_return,
#                             "net_return": net_return,
#                             "hold_bars": hold_bars,
#                             "forced_exit": force_close,
#                         }
#                     )

#                     last_equity = last_equity * (1.0 + net_return)
#                     equity_changed_this_step = True

#                     # Reset position, start cooldown
#                     position = 0
#                     entry_idx = None
#                     entry_price = None
#                     cooldown = self.cooldown_steps

#             # Append equity for this step
#             if not equity_changed_this_step:
#                 # No trade closed → carry forward
#                 equity_curve.append(last_equity)
#             else:
#                 equity_curve.append(last_equity)

#         results = {
#             "equity_curve": np.array(equity_curve),
#             "trades": trades,
#         }
#         return results
    
#     def _rolling_mean(self, arr: np.ndarray, window: int) -> np.ndarray:
#         if window <= 1:
#             return arr.copy()
#         cumsum = np.cumsum(np.insert(arr, 0, 0.0))
#         out = (cumsum[window:] - cumsum[:-window]) / float(window)
#         # pad to same length
#         pad = np.full(window - 1, np.nan, dtype=np.float32)
#         return np.concatenate([pad, out])

import numpy as np
from typing import List, Dict, Any

from config.system import TradingSystemConfig
from agents.ddqn import DDQNAgent


class TradeManager:
    """
    Orchestrates BuyAgent and SellAgent to execute full trades
    on historical data in a single forward pass.
    """

    def __init__(
        self,
        buy_agent: DDQNAgent,
        sell_agent: DDQNAgent,
        state_df: np.ndarray,
        prices: np.ndarray,
        config: TradingSystemConfig,
    ):
        self.buy_agent = buy_agent
        self.sell_agent = sell_agent
        self.state_df = state_df
        self.prices = prices
        self.config = config

        self.n_steps = len(prices)

        # Trading state
        self.position_open = False
        self.entry_idx = None
        self.entry_price = None
        self.hold_bars = 0
        self.cooldown = 0

        # Accounting
        self.equity = 1.0
        self.equity_curve = []
        self.trades: List[Dict[str, Any]] = []

    # --------------------------------------------------
    # Main execution
    # --------------------------------------------------

    def run(self) -> Dict[str, Any]:
        for t in range(self.n_steps):

            state = self.state_df[t]
            price = self.prices[t]

            # -----------------------
            # Cooldown
            # -----------------------
            if self.cooldown > 0:
                self.cooldown -= 1
                self._record_equity()
                continue

            # -----------------------
            # BUY decision (flat)
            # -----------------------
            if not self.position_open:
                buy_action = self.buy_agent.select_action(state, greedy=True)

                if buy_action == 1:
                    self._open_position(t, price)

            # -----------------------
            # SELL decision (long)
            # -----------------------
            else:
                self.hold_bars += 1
                sell_action = self.sell_agent.select_action(state, greedy=True)

                forced = self.hold_bars >= self.config.trade_manager.sell_horizon

                if sell_action == 1 or forced:
                    self._close_position(t, price, forced)

            self._record_equity()

        return {
            "trades": self.trades,
            "equity_curve": np.array(self.equity_curve),
            "final_equity": self.equity,
        }

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _open_position(self, t: int, price: float):
        self.position_open = True
        self.entry_idx = t
        self.entry_price = price
        self.hold_bars = 0

    def _close_position(self, t: int, price: float, forced: bool):
        gross_return = (price - self.entry_price) / self.entry_price
        net_return = gross_return - self.config.trade_manager.transaction_cost

        self.equity *= (1.0 + net_return)

        self.trades.append({
            "entry_idx": self.entry_idx,
            "exit_idx": t,
            "entry_price": self.entry_price,
            "exit_price": price,
            "gross_return": gross_return,
            "net_return": net_return,
            "hold_bars": self.hold_bars,
            "forced_exit": forced,
        })

        # Reset position
        self.position_open = False
        self.entry_idx = None
        self.entry_price = None
        self.hold_bars = 0
        self.cooldown = self.config.trade_manager.cooldown_steps

    def _record_equity(self):
        self.equity_curve.append(self.equity)
