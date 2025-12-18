import yaml
from config.system import TradingSystemConfig
from config.data import DataConfig
from config.feature import FeatureConfig
from config.state import StateConfig
from config.scaling import ScalingConfig
from config.reward import RewardConfig
from config.agents import AgentConfig
from config.training import TrainingConfig
from config.trade_manager import TradeManagerConfig

def load_config(path: str) -> TradingSystemConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return TradingSystemConfig(
        data=DataConfig(
            tickers=raw["tickers"],
            start_date=raw["start_date"],
            end_date=raw["end_date"],
            interval=raw["interval"],
        ),
        features=FeatureConfig(
            technical_indicators=raw["technical_indicators"],
            include_fundamentals=raw["include_fundamentals"],
            include_sentiment=raw["include_sentiment"],
        ),
        state=StateConfig(
            window_size=raw["state"]["window_size"]
        ),
        scaling=ScalingConfig(**raw["scaling"]),
        reward=RewardConfig(
            transaction_cost=raw["reward"]["transaction_cost"]
        ),
        agent=AgentConfig(
            state_dim=None,   # injected later after feature build
            n_actions=2,
        ),
        training=TrainingConfig(
            episodes=200,
            warmup_steps=238,
            multiprocess=True,
        ),
        trade_manager=TradeManagerConfig(**raw["trade_manager"]),
    )
