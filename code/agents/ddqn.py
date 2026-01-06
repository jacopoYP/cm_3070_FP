import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """
    Simple MLP that maps state -> Q-values for each action.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition.
        """
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

# ---------------------------
# DDQN Agent
# ---------------------------

from collections import deque
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """
    Simple feed-forward network for Q-values.
    Input: state vector
    Output: Q(s, a) for each action
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_sizes=(128, 128)):
        super().__init__()

        layers = []
        last_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, n_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """
    Basic replay buffer using deque (maxlen = capacity).
    Stores tuples: (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.stack(states).astype(np.float32)
        next_states = np.stack(next_states).astype(np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


from config.agents import AgentConfig

class DDQNAgent:
    def __init__(self, cfg: AgentConfig, device: str | None = None):
        self.cfg = cfg

        self.state_dim = cfg.state_dim
        self.n_actions = cfg.n_actions
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.target_update_freq = cfg.target_update_freq
        self.loss_history = []

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.q_net = QNetwork(cfg.state_dim, cfg.n_actions).to(self.device)
        self.target_net = QNetwork(cfg.state_dim, cfg.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(cfg.buffer_size)

        # Epsilon schedule
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay_steps = cfg.epsilon_decay_steps
        self.epsilon = cfg.epsilon_start
        self.total_steps = 0

        self.learn_step = 0

        self.loss_history = []

    # --------------------------------------------------
    # Policy
    # --------------------------------------------------

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy:
            self._anneal_epsilon()

        if (not greedy) and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return torch.argmax(self.q_net(s), dim=1).item()

    def _anneal_epsilon(self):
        self.total_steps += 1
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        q_sa = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = torch.argmax(self.q_net(next_states), dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.gamma * (1 - dones) * next_q

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.loss_history.append(loss.item())
        return loss.item()
    
    # Replay Buffer Interface
    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q(s, :) as a numpy array."""
        if not isinstance(state, np.ndarray):
            state = np.asarray(state, dtype=np.float32)
        state = state.astype(np.float32)

        with torch.no_grad():
            s = torch.from_numpy(state).to(self.device).unsqueeze(0)
            q = self.q_net(s).squeeze(0).detach().cpu().numpy()
        return q
    
    # def act_with_confidence(self, state: np.ndarray) -> tuple[int, float]:
    #     """
    #     Returns:
    #         action: int
    #         confidence: float in [0, 1]
    #     """
    #     if not isinstance(state, np.ndarray):
    #         state = np.asarray(state, dtype=np.float32)

    #     state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

    #     with torch.no_grad():
    #         q_values = self.q_net(state).cpu().numpy().squeeze()

    #     action = int(np.argmax(q_values))

    #     # Confidence definition (stable, bounded)
    #     q_max = float(np.max(q_values))
    #     q_mean = float(np.mean(q_values))

    #     confidence = q_max - q_mean

    #     # Optional: squash to [0, 1] for interpretability
    #     confidence = 1.0 / (1.0 + np.exp(-confidence))

    #     return action, confidence
    def act_with_confidence(self, state: np.ndarray, buy_index: int = 1, temp: float = 1.0) -> tuple[int, float]:
        if not isinstance(state, np.ndarray):
            state = np.asarray(state, dtype=np.float32)

        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(s).cpu().numpy().squeeze()

        action = int(np.argmax(q))

        # BUY advantage margin vs best alternative
        q_buy = float(q[buy_index])
        q_other = float(np.max(np.delete(q, buy_index))) if len(q) > 1 else q_buy
        margin = (q_buy - q_other) / max(1e-8, temp)


        conf = 1.0 / (1.0 + np.exp(-margin))  # sigmoid to [0,1]
        return action, float(conf)





    # --------------------------------------------------
    # GA support
    # --------------------------------------------------

    def clone(self):
        """
        Create a fresh agent with same hyperparameters (no weights).
        """
        return DDQNAgent(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            gamma=self.gamma,
            lr=self.lr,
            batch_size=self.batch_size,
            buffer_size=self.buffer_size,
            target_update_freq=self.target_update_freq,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_steps=self.epsilon_decay_steps,
            device=str(self.device),
        )
