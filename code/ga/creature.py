# ga/creature.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from ga.genome import Genome


@dataclass
class Creature:
    genome: Genome
    fitness: Optional[float] = None
    metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genes": dict(self.genome.values),
            "fitness": self.fitness,
            "metrics": dict(self.metrics),
        }
