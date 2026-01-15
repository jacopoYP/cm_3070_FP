from dataclasses import dataclass

@dataclass(frozen=True)
class TradeManagerConfig:
    cooldown_steps: int
    sell_horizon: int
    confidence_temp: float
    buy_min_confidence: float
    use_trend_filter: bool
    ma_short: int
    ma_long: int
    min_hold_bars: int