import numpy as np

class AlwaysHoldSellAgent:
    """Sell stub used ONLY for Buy checkpoint evaluation.
    It never triggers SELL, so TradeManager exits via time-stop (sell_horizon)."""
    def select_action(self, state: np.ndarray, greedy: bool = True) -> int:
        return 0
