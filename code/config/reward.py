from dataclasses import dataclass

@dataclass(frozen=True)
class RewardConfig:
    transaction_cost: float
    lambda_dd: float = 0.05
    lambda_vol: float = 0.01
    hold_penalty_long: float = 0.0
