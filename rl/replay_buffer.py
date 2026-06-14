"""
Experience Replay Buffer for online RL training.

Supports:
- Uniform random sampling (standard replay buffer)
- Prioritized Experience Replay (PER) with proportional priorities
- Save/Load buffer to disk for resuming training
"""

import random
import os
import numpy as np
import torch
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional

from utils.my_logger import logger


@dataclass
class Transition:
    """A single experience transition."""
    state: np.ndarray       # shape: (frames, H, W), float32, [0,1]
    action: int
    reward: float
    next_state: np.ndarray  # shape: (frames, H, W), float32, [0,1]
    done: bool


class RunningRewardNormalizer:
    """Online running mean/std normalizer for rewards (Welford's algorithm)."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, reward: float):
        self.count += 1
        if self.count == 1:
            self.mean = reward
            self.var = 0.0
        else:
            delta = reward - self.mean
            self.mean += delta / self.count
            delta2 = reward - self.mean
            self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, reward: float) -> float:
        std = max(self.var ** 0.5, self.eps)
        return (reward - self.mean) / std

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict):
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


class ReplayBuffer:
    """
    Standard uniform replay buffer using a ring buffer (deque).

    Usage:
        buffer = ReplayBuffer(capacity=100000)
        buffer.push(state, action, reward, next_state, done)
        if len(buffer) >= min_samples:
            batch = buffer.sample(batch_size)
    """

    def __init__(self, capacity: int = 100000, normalize_reward: bool = True):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.normalize_reward = normalize_reward
        self.reward_normalizer = RunningRewardNormalizer() if normalize_reward else None

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Store a transition in the buffer (reward is normalized if enabled)."""
        if self.reward_normalizer is not None:
            self.reward_normalizer.update(reward)
            reward = self.reward_normalizer.normalize(reward)
        self.buffer.append(Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        ))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of transitions.

        Returns:
            (states, actions, rewards, next_states, dones) as tensors
            states/next_states shape: (batch, frames, H, W)
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = torch.from_numpy(np.stack([t.state for t in batch])).float()
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).float()
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_samples: int) -> bool:
        """Check if buffer has enough samples to start training."""
        return len(self.buffer) >= min_samples

    def clear(self):
        self.buffer.clear()

    def save(self, path: str):
        """Save buffer contents to disk."""
        data = {
            "capacity": self.capacity,
            "normalize_reward": self.normalize_reward,
            "reward_normalizer": self.reward_normalizer.state_dict() if self.reward_normalizer else None,
            "transitions": [
                {
                    "state": t.state,
                    "action": t.action,
                    "reward": t.reward,
                    "next_state": t.next_state,
                    "done": t.done,
                }
                for t in self.buffer
            ],
        }
        tmp_path = path + ".tmp"
        np.savez_compressed(tmp_path, data=np.array([data], dtype=object))
        os.replace(tmp_path, path)
        logger.info(f"ReplayBuffer saved: {len(self.buffer)} transitions -> {path}")

    def load(self, path: str):
        """Load buffer contents from disk."""
        loaded = np.load(path, allow_pickle=True)
        data = loaded["data"][0]
        self.capacity = data["capacity"]
        self.buffer = deque(maxlen=self.capacity)
        # restore normalizer state
        self.normalize_reward = data.get("normalize_reward", self.normalize_reward)
        if self.normalize_reward and data.get("reward_normalizer"):
            if self.reward_normalizer is None:
                self.reward_normalizer = RunningRewardNormalizer()
            self.reward_normalizer.load_state_dict(data["reward_normalizer"])
        for t in data["transitions"]:
            self.buffer.append(Transition(
                state=t["state"],
                action=t["action"],
                reward=float(t["reward"]),
                next_state=t["next_state"],
                done=t["done"],
            ))
        logger.info(f"ReplayBuffer loaded: {len(self.buffer)} transitions from {path}")


class SumTree:
    """Binary sum-tree for efficient prioritized sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.

    Higher-error transitions are sampled more frequently.
    Uses proportional prioritization with importance sampling weights.

    Usage:
        buffer = PrioritizedReplayBuffer(capacity=100000)
        buffer.push(state, action, reward, next_state, done)
        if buffer.is_ready(min_samples):
            batch, indices, weights = buffer.sample(batch_size, beta=0.4)
            # After computing TD errors:
            buffer.update_priorities(indices, td_errors)
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_anneal_steps: int = 100000, epsilon: float = 1e-6,
                 normalize_reward: bool = True):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling correction
            beta_end: Final beta value (annealed over beta_anneal_steps)
            beta_anneal_steps: Number of sampling steps to anneal beta
            epsilon: Small constant added to priorities to avoid zero probability
            normalize_reward: Whether to normalize rewards with running stats
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.sample_count = 0
        self.normalize_reward = normalize_reward
        self.reward_normalizer = RunningRewardNormalizer() if normalize_reward else None

    def _current_beta(self) -> float:
        progress = min(1.0, self.sample_count / max(1, self.beta_anneal_steps))
        return self.beta_start + (self.beta_end - self.beta_start) * progress

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Store transition with max priority (ensures new experiences get sampled)."""
        if self.reward_normalizer is not None:
            self.reward_normalizer.update(reward)
            reward = self.reward_normalizer.normalize(reward)
        transition = Transition(state=state, action=action, reward=reward,
                                next_state=next_state, done=done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        np.ndarray,
        torch.Tensor,
    ]:
        """
        Sample a batch with prioritized probabilities.

        Returns:
            (states, actions, rewards, next_states, dones), tree_indices, importance_weights
        """
        self.sample_count += 1
        beta = self._current_beta()

        batch_size = min(batch_size, self.tree.size)
        indices = np.zeros(batch_size, dtype=np.int64)
        transitions = []
        priorities = np.zeros(batch_size, dtype=np.float64)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s = random.uniform(low, high)
            idx, priority, data = self.tree.get(s)
            indices[i] = idx
            priorities[i] = priority
            transitions.append(data)

        # Importance sampling weights
        total = self.tree.total
        min_prob = priorities.min() / total
        max_weight = (min_prob * self.tree.size) ** (-beta)

        weights = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            prob = priorities[i] / total
            weight = (prob * self.tree.size) ** (-beta)
            weights[i] = weight / max_weight

        states = torch.from_numpy(np.stack([t.state for t in transitions])).float()
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([t.next_state for t in transitions])).float()
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32)

        return (states, actions, rewards, next_states, dones), indices, torch.from_numpy(weights)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, abs(td_error) + self.epsilon)

    def __len__(self) -> int:
        return self.tree.size

    def is_ready(self, min_samples: int) -> bool:
        return self.tree.size >= min_samples

    def clear(self):
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self.sample_count = 0
