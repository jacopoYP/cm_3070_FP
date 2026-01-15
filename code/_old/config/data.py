from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class DataConfig:
    tickers: List[str]
    start_date: str
    end_date: str
    interval: str
