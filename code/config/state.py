# config/state.py
from dataclasses import dataclass

@dataclass(frozen=True)
class StateConfig:
    window_size: int = 30
