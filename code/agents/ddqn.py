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


class DDQNAgent:
    """
    Double DQN agent:
    - q_net: online network
    - target_net: target network
    - epsilon-greedy exploration
    - replay buffer + mini-batch updates
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 20_000,
        target_update_freq: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5_000,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.loss_history = []

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.q_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Epsilon-greedy schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.total_steps = 0  # used to anneal epsilon

        # Internal training step counter
        self.learn_step = 0

    # Policy
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        - state: np.array (state_dim,)
        - if greedy=True -> always exploit (used for evaluation)
        """
        if not isinstance(state, np.ndarray):
            state = np.asarray(state, dtype=np.float32)
        state = state.astype(np.float32)

        # Update epsilon based on total steps (except in greedy mode)
        if not greedy:
            self.total_steps += 1
            frac = min(1.0, self.total_steps / float(self.epsilon_decay_steps))
            self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

        if (not greedy) and (random.random() < self.epsilon):
            # Explore
            return random.randrange(self.n_actions)

        # Exploit
        with torch.no_grad():
            s = torch.from_numpy(state).to(self.device).unsqueeze(0)
            q_values = self.q_net(s)
            action = torch.argmax(q_values, dim=1).item()
        return action

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

    # Training Step (DDQN Update)
    def update(self) -> float | None:
        """
        One DDQN update step.
        Returns the loss value (float) or None if not enough data yet.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Same values we encountered for the Atari breakout agent training
        states = torch.from_numpy(states).to(self.device)          
        next_states = torch.from_numpy(next_states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        

        # Q(s,a) for taken actions
        q_values = self.q_net(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # DDQN: action selection from q_net, evaluation from target_net
        with torch.no_grad():
            next_q_online = self.q_net(next_states)
            next_actions = torch.argmax(next_q_online, dim=1)

            next_q_target = self.target_net(next_states)
            next_q_sa = next_q_target.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            target = rewards + self.gamma * (1.0 - dones) * next_q_sa

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Increase learn_step + update target network periodically
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.loss_history.append(loss.item())

        return loss.item()