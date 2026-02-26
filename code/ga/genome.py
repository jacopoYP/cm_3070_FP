# ga/genome.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import random


@dataclass(frozen=True)
class GeneSpec:
    """
    One tunable hyperparameter.
    - name: label used in logs
    - path: dotted path to config (e.g., "trade_manager.buy_min_confidence")
    - kind: "float" | "int" | "categorical"
    - bounds: (low, high) for float/int
    - choices: list of values for categorical
    """
    name: str
    path: str
    kind: str = "float"
    bounds: Tuple[float, float] | None = None
    choices: List[Any] | None = None

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "float":
            lo, hi = self.bounds
            return float(rng.uniform(float(lo), float(hi)))
        if self.kind == "int":
            lo, hi = self.bounds
            return int(rng.randint(int(lo), int(hi)))
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"categorical gene {self.name} has no choices")
            return rng.choice(self.choices)
        raise ValueError(f"Unknown gene kind: {self.kind}")

    def clamp(self, v: Any) -> Any:
        if self.kind == "float":
            lo, hi = self.bounds
            return float(min(max(float(v), float(lo)), float(hi)))
        if self.kind == "int":
            lo, hi = self.bounds
            return int(min(max(int(v), int(lo)), int(hi)))
        if self.kind == "categorical":
            return v
        return v


class Genome:
    # A genome is a mapping {gene_name: value}, with the gene specs stored separately.
    
    def __init__(self, genes: List[GeneSpec], values: Dict[str, Any]):
        self.genes = genes
        self.values = dict(values)

    @classmethod
    def random(cls, genes: List[GeneSpec], rng: random.Random) -> "Genome":
        vals = {g.name: g.sample(rng) for g in genes}
        return cls(genes, vals)

    def as_config_overrides(self) -> Dict[str, Any]:
        out = {}
        for g in self.genes:
            out[g.path] = self.values[g.name]
        return out

    def copy(self) -> "Genome":
        return Genome(self.genes, dict(self.values))

    def mutate(self, rng: random.Random, p_mut: float, sigma: float = 0.05) -> "Genome":
        #  gaussian mutation within bounds

        child = self.copy()
        for g in self.genes:
            if rng.random() > p_mut:
                continue

            cur = child.values[g.name]

            if g.kind == "float":
                lo, hi = g.bounds  # type: ignore[misc]
                span = float(hi) - float(lo)
                
                # sigma is relative to span
                step = rng.gauss(0.0, sigma * span)
                child.values[g.name] = g.clamp(float(cur) + step)

            elif g.kind == "int":
                step = rng.choice([-2, -1, 1, 2])
                child.values[g.name] = g.clamp(int(cur) + step)

            elif g.kind == "categorical":
                child.values[g.name] = g.sample(rng)

        return child

    @staticmethod
    def crossover(a: "Genome", b: "Genome", rng: random.Random) -> "Genome":
        # Uniform crossover: for each gene I pick from parent A or B
        
        if a.genes != b.genes:
            raise ValueError("Cannot crossover genomes with different gene specs")

        vals = {}
        for g in a.genes:
            vals[g.name] = a.values[g.name] if rng.random() < 0.5 else b.values[g.name]
        return Genome(a.genes, vals)
