from dataclasses import dataclass

@dataclass
class ScalingConfig:
    window: int
    method: str  # e.g. "rolling_zscore"
