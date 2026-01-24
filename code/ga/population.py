# ga/population.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple
import random

from ga.genome import GeneSpec, Genome
from ga.creature import Creature


@dataclass
class GAConfig:
    population_size: int = 16
    generations: int = 10
    elite_frac: float = 0.25     # keep top X%
    tournament_k: int = 3
    crossover_rate: float = 0.6
    mutation_rate: float = 0.3
    mutation_sigma: float = 0.05
    seed: int = 42


class Population:
    def __init__(self, genes: List[GeneSpec], cfg: GAConfig, evaluator: Callable[[Genome], Tuple[float, dict]]):
        self.genes = genes
        self.cfg = cfg
        self.evaluator = evaluator
        self.rng = random.Random(int(cfg.seed))
        self.creatures: List[Creature] = []

    def init(self) -> None:
        self.creatures = [
            Creature(genome=Genome.random(self.genes, self.rng))
            for _ in range(int(self.cfg.population_size))
        ]

    def evaluate_all(self) -> None:
        for c in self.creatures:
            if c.fitness is not None:
                continue
            fit, metrics = self.evaluator(c.genome)
            c.fitness = float(fit)
            c.metrics = dict(metrics)

    def best(self) -> Creature:
        self.evaluate_all()
        return max(self.creatures, key=lambda c: float(c.fitness))

    def evolve_one_generation(self) -> Creature:
        self.evaluate_all()
        self.creatures.sort(key=lambda c: float(c.fitness), reverse=True)

        elite_n = max(1, int(self.cfg.elite_frac * len(self.creatures)))
        elites = self.creatures[:elite_n]

        next_gen: List[Creature] = [Creature(genome=e.genome.copy(), fitness=e.fitness, metrics=dict(e.metrics)) for e in elites]

        while len(next_gen) < len(self.creatures):
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            if self.rng.random() < float(self.cfg.crossover_rate):
                child_genome = Genome.crossover(p1.genome, p2.genome, self.rng)
            else:
                child_genome = p1.genome.copy()

            child_genome = child_genome.mutate(
                self.rng,
                p_mut=float(self.cfg.mutation_rate),
                sigma=float(self.cfg.mutation_sigma),
            )

            next_gen.append(Creature(genome=child_genome))

        self.creatures = next_gen[: len(self.creatures)]
        return self.best()

    def _tournament_select(self) -> Creature:
        k = max(2, int(self.cfg.tournament_k))
        contenders = [self.creatures[self.rng.randrange(len(self.creatures))] for _ in range(k)]
        return max(contenders, key=lambda c: float(c.fitness))
