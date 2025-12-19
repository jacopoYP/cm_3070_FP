import yaml
import numpy as np
import torch

from agents.buy_agent_trainer import BuyAgentTrainer
from agents.sell_agent_trainer import SellAgentTrainer

from config.system import TradingSystemConfig

from features.state_assembler import StateAssembler


class TradeManager:
    def __init__(
        self,
        buy_agent,
        sell_agent,
        state_df,
        prices,
        # feature_cols: np.ndarray,
        config: TradingSystemConfig,
    ):
        self.buy_agent = buy_agent
        self.sell_agent = sell_agent
        self.state_df = state_df
        self.prices = prices
        # self.feature_cols = feature_cols
        self.config = config

        self.trades = []
        self.equity_curve = []

        # self.assembler = StateAssembler(
        #     feature_cols=self.feature_cols,
        #     # window_size=self.config.agent.window_size,
        #     window_size=self.config.state.window_size,
        # )

    # def run(self):
    #     in_position = False
    #     entry_idx = None
    #     entry_price = None
    #     equity = 1.0
    #     cooldown = 0

    #     # window_size = self.config.state.window_size

    #     # for t in range(window_size - 1, len(self.prices) - 1):
    #     # state = self.state_df[t]

    #     for t in range(len(self.prices) - 1):
    #         state = self.state_df.iloc[t].values.astype(np.float32)

    #         assert state.ndim == 1
    #         assert state.shape[0] == self.buy_agent.state_dim

    #         # state.shape == (270,)


    #         # -------------------
    #         # COOLDOWN
    #         # -------------------
    #         if cooldown > 0:
    #             cooldown -= 1
    #             self.equity_curve.append(equity)
    #             continue

    #         # -------------------
    #         # ENTRY LOGIC
    #         # -------------------
    #         if not in_position:
    #             buy_action, buy_conf = self.buy_agent.act_with_confidence(state)

    #             if buy_action == 1 and buy_conf >= self.config.trade_manager.buy_min_confidence:
    #                 in_position = True
    #                 entry_idx = t
    #                 entry_price = self.prices[t]
    #                 continue

    #         # -------------------
    #         # EXIT LOGIC
    #         # -------------------
    #         if in_position:
    #             sell_state = self.state_df[t]
    #             sell_action = self.sell_agent.select_action(sell_state, greedy=True)

    #             hold_bars = t - entry_idx
    #             force_exit = hold_bars >= self.config.trade_manager.sell_horizon

    #             if sell_action == 1 or force_exit:
    #                 exit_price = self.prices[t]
    #                 ret = (exit_price - entry_price) / entry_price
    #                 ret -= self.config.reward.transaction_cost

    #                 equity *= (1.0 + ret)

    #                 self.trades.append({
    #                     "entry_idx": entry_idx,
    #                     "exit_idx": t,
    #                     "return": ret,
    #                     "forced_exit": force_exit,
    #                 })

    #                 in_position = False
    #                 cooldown = self.config.trade_manager.cooldown_steps

    #         self.equity_curve.append(equity)

    #     return {
    #         "final_equity": equity,
    #         "trades": self.trades,
    #         "equity_curve": self.equity_curve,
    #     }
    import numpy as np

    def run(self):
        equity = 1.0
        equity_curve = []
        trades = []

        in_position = False
        entry_price = None
        entry_idx = None
        cooldown = 0

        for t in range(len(self.prices) - 1):

            # ✅ CORRECT state extraction
            state = self.state_df.iloc[t].values.astype(np.float32)

            # HARD invariants
            assert state.ndim == 1
            assert state.shape[0] == self.buy_agent.state_dim

            # price = self.prices[t]
            price = self.prices.iloc[t]

            # -------------------
            # ENTRY LOGIC
            # -------------------
            if not in_position and cooldown == 0:
                # action = self.buy_agent.act(state)
                action = self.buy_agent.select_action(state, greedy=True)

                if action == 1:
                    in_position = True
                    entry_price = price
                    entry_idx = t

            # -------------------
            # EXIT LOGIC
            # -------------------
            elif in_position:
                # action = self.sell_agent.act(state)
                action = self.sell_agent.select_action(state, greedy=True)

                if action == 1:
                    exit_price = price
                    gross_return = (exit_price - entry_price) / entry_price
                    net_return = gross_return - self.config.reward.transaction_cost

                    equity *= (1.0 + net_return)

                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": t,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "net_return": net_return,
                    })

                    in_position = False
                    cooldown = self.config.trade_manager.cooldown_steps

            # -------------------
            # COOLDOWN
            # -------------------
            if cooldown > 0:
                cooldown -= 1

            equity_curve.append(equity)

        return {
            "final_equity": equity,
            "equity_curve": equity_curve,
            "trades": trades,
        }

