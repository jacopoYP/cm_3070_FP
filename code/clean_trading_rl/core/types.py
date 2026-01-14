from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =========================
# Config dataclasses
# =========================


@dataclass
class DataConfig:
    tickers: List[str] = field(default_factory=list)
    start_date: str = "2018-01-01"
    end_date: str = "2024-01-01"
    interval: str = "1d"


@dataclass
class FeaturesConfig:
    technical_indicators: List[str] = field(default_factory=list)
    include_fundamentals: bool = False
    include_sentiment: bool = False
    scaling: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardConfig:
    transaction_cost: float = 0.001
    lambda_dd: float = 0.0
    lambda_vol: float = 0.0

    # Minimal shaping knobs (safe defaults)
    flat_hold_penalty: float = 0.0       # penalty when flat and HOLD
    in_pos_hold_penalty: float = 0.0     # penalty when in position and HOLD
    entry_bonus: float = 0.0             # bonus on opening a position


@dataclass
class StateConfig:
    window_size: int = 30


@dataclass
class AgentConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 200_000
    target_update_freq: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 40_000

    # filled at runtime
    state_dim: int = 0
    n_actions: int = 2


@dataclass
class TrainingConfig:
    episodes: int = 200
    warmup_steps: int = 200
    steps_per_episode: Optional[int] = None  # None means run until env done
    log_every: int = 1
    seed: int = 42


@dataclass
class TradeManagerConfig:
    cooldown_steps: int = 5
    sell_horizon: int = 20
    # min_hold_bars: int = 2
    min_hold_bars: int = 10

    # Confidence gating
    confidence_method: str = "margin_sigmoid"  # softmax | margin_sigmoid
    confidence_temp: float = 0.02
    # buy_min_confidence: float = 0.51
    buy_min_confidence: float = 0.40
    sell_min_confidence: float = 0.0

    # Optional trend filter
    use_trend_filter: bool = False
    ma_short: int = 10
    ma_long: int = 30


@dataclass
class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    state: StateConfig = field(default_factory=StateConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trade_manager: TradeManagerConfig = field(default_factory=TradeManagerConfig)
