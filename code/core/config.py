from __future__ import annotations

import yaml
from dataclasses import asdict
from typing import Any, Dict

from .types import (
    AgentConfig,
    DataConfig,
    FeaturesConfig,
    RewardConfig,
    StateConfig,
    SystemConfig,
    AlphaVantageConfig,
    TradeManagerConfig,
    TrainingConfig,
)


def _merge_dataclass(dc, overrides: Dict[str, Any]):
    """Shallow merge: only known fields in overrides are applied."""
    for k, v in overrides.items():
        if hasattr(dc, k):
            setattr(dc, k, v)
    return dc

def load_config(path: str) -> SystemConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = SystemConfig()

    # Allow top-level (legacy) keys OR nested keys.
    # If you later want strictness, we can enforce nesting only.
    cfg.data = _merge_dataclass(cfg.data, raw.get("data", raw))
    cfg.features = _merge_dataclass(cfg.features, raw.get("features", raw))
    cfg.reward = _merge_dataclass(cfg.reward, raw.get("reward", raw))
    cfg.state = _merge_dataclass(cfg.state, raw.get("state", raw))
    cfg.agent = _merge_dataclass(cfg.agent, raw.get("agent", raw))
    cfg.training = _merge_dataclass(cfg.training, raw.get("training", raw))
    cfg.trade_manager = _merge_dataclass(cfg.trade_manager, raw.get("trade_manager", raw))
    cfg.alphavantage = _merge_dataclass(cfg.alphavantage, raw.get("alphavantage", raw))

    return cfg

def save_config(cfg: SystemConfig, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
