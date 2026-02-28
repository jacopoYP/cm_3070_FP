from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.types import AgentConfig
from agents.networks import MLPQNetwork
from agents.replay_buffer import ReplayBuffer


class DDQNAgent:
    # Double DQN with replay buffer.

    def __init__(self, cfg: AgentConfig, device: Optional[str] = None, hidden_sizes=(128, 128)):
        self.cfg = cfg
        self.state_dim = int(cfg.state_dim)
        self.n_actions = int(cfg.n_actions)

        if self.state_dim <= 0:
            raise ValueError("cfg.state_dim must be set before creating DDQNAgent")

        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        )

        self.q = MLPQNetwork(self.state_dim, self.n_actions, hidden_sizes=hidden_sizes).to(self.device)
        self.q_target = MLPQNetwork(self.state_dim, self.n_actions, hidden_sizes=hidden_sizes).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=float(cfg.lr))
        self.loss_fn = nn.SmoothL1Loss()

        self.buf = ReplayBuffer(int(cfg.buffer_size))

        self.eps = float(cfg.epsilon_start)
        self._steps = 0
        self.loss_history: list[float] = []
        self.learn_steps = 0

        self.total_steps = 0

    # ---------- public API ----------

    def select_action(self, state, greedy: bool = False) -> int:
        if not greedy:
            self.total_steps += 1
            self._update_epsilon()

        if (not greedy) and (np.random.rand() < self.eps):
            return np.random.randint(self.cfg.n_actions)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(s)
            return int(torch.argmax(q, dim=1).item())

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qv = self.q(s)[0].detach().cpu().numpy()
        return qv

    def push(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool) -> None:
        self.buf.push(s, a, r, ns, done)

    def update(self):
        if len(self.buf) < int(self.cfg.batch_size):
            return None

        s, a, r, ns, d = self.buf.sample(int(self.cfg.batch_size))

        s  = torch.tensor(np.asarray(s),  dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.asarray(ns), dtype=torch.float32, device=self.device)
        a  = torch.tensor(np.asarray(a),  dtype=torch.int64,   device=self.device)
        r  = torch.tensor(np.asarray(r),  dtype=torch.float32, device=self.device)
        d  = torch.tensor(np.asarray(d),  dtype=torch.float32, device=self.device).clamp(0.0, 1.0)

        q_out = self.q(s)

        # convert inference tensor -> normal tensor
        if q_out.is_inference():
            q_out = q_out.clone()

        q_sa = q_out.gather(1, a.unsqueeze(1)).squeeze(1)

        # DDQN target (no grad)
        with torch.no_grad():
            # online selects action
            q_next_online = self.q(ns)
            if q_next_online.is_inference():
                q_next_online = q_next_online.clone()
            next_a = torch.argmax(q_next_online, dim=1)

            # target evaluates that action
            q_next_target = self.q_target(ns)
            if q_next_target.is_inference():
                q_next_target = q_next_target.clone()
            next_q = q_next_target.gather(1, next_a.unsqueeze(1)).squeeze(1)

            target = r + float(self.cfg.gamma) * (1.0 - d) * next_q

        loss = self.loss_fn(q_sa, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

        self.learn_steps += 1
        val = float(loss.detach().cpu().item())
        self.loss_history.append(val)

        if self.learn_steps % int(self.cfg.target_update_freq) == 0:
            # print("target sync @", self.learn_steps, "loss", val)
            self.q_target.load_state_dict(self.q.state_dict())

        return val

    def save(self, path: str) -> None:
        payload = {
            "cfg": asdict(self.cfg),
            "state_dict": self.q.state_dict(),
            "target_state_dict": self.q_target.state_dict(),
            "eps": self.eps,
            "steps": self._steps,
            "learn_steps": self.learn_steps,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.q.load_state_dict(payload["state_dict"])
        self.q_target.load_state_dict(payload.get("target_state_dict", payload["state_dict"]))
        self.eps = float(payload.get("eps", self.eps))
        self._steps = int(payload.get("steps", self._steps))
        self.learn_steps = int(payload.get("learn_steps", self.learn_steps))

    # ---------------------------------------------------------------------
    def _update_epsilon(self) -> None:
        decay_steps = float(self.cfg.epsilon_decay_steps)
        if decay_steps <= 0:
            self.eps = float(self.cfg.epsilon_end)
            return

        frac = min(1.0, self.total_steps / decay_steps)
        self.eps = float(self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start))


