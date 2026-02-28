from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np

# Standard Replay buffer
class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self._buf.append((state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(ns).astype(np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)
