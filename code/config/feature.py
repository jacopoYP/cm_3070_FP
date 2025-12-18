from dataclasses import dataclass
from typing import List

@dataclass
class FeatureConfig:
    technical_indicators: List[str]
    include_fundamentals: bool
    include_sentiment: bool
