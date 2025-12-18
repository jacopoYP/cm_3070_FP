from dataclasses import dataclass
from config.data import DataConfig
from config.feature import FeatureConfig
from config.scaling import ScalingConfig
from config.reward import RewardConfig
from config.agents import AgentConfig
from config.state import StateConfig
from config.training import TrainingConfig
from config.trade_manager import TradeManagerConfig

@dataclass(frozen=True)
class TradingSystemConfig:
    data: DataConfig
    state: StateConfig
    features: FeatureConfig
    scaling: ScalingConfig
    reward: RewardConfig
    agent: AgentConfig
    training: TrainingConfig
    trade_manager: TradeManagerConfig
