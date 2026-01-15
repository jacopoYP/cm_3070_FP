from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingConfig:
    episodes: int
    warmup_steps: int
    multiprocess: bool
    n_workers: int = 4
    steps_per_batch: int = 200
    updates_per_batch: int = 20
