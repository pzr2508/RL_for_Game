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
    Numpy-backed ring buffer for memory-efficient replay.

    Stores frames as a single contiguous numpy array instead of a deque of
    Transition objects, reducing memory overhead by ~3x and making
    save/load significantly faster (single array dump vs per-transition serialization).

    Usage:
        buffer = ReplayBuffer(capacity=100000, state_shape=(10, 84, 84))
        buffer.push(state, action, reward, next_state, done)
        if buffer.is_ready(min_samples):
            batch = buffer.sample(batch_size)
    """

    def __init__(self, capacity: int = 100000, normalize_reward: bool = True,
                 state_shape: Tuple[int, ...] = (10, 84, 84)):
        self.capacity = capacity
        self.normalize_reward = normalize_reward
        self.reward_normalizer = RunningRewardNormalizer() if normalize_reward else None

        # Pre-allocate contiguous arrays — avoids per-push allocation
        self._states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._write_idx = 0
        self._size = 0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Store a transition (reward is normalized if enabled)."""
        if self.reward_normalizer is not None:
            self.reward_normalizer.update(reward)
            reward = self.reward_normalizer.normalize(reward)

        idx = self._write_idx
        self._states[idx] = state
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_states[idx] = next_state
        self._dones[idx] = float(done)

        self._write_idx = (self._write_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of transitions.

        Returns:
            (states, actions, rewards, next_states, dones) as tensors.
            states/next_states shape: (batch, frames, H, W)
        """
        n = min(batch_size, self._size)
        indices = np.random.randint(0, self._size, size=n)

        states = torch.from_numpy(self._states[indices].copy())
        actions = torch.from_numpy(self._actions[indices].copy())
        rewards = torch.from_numpy(self._rewards[indices].copy())
        next_states = torch.from_numpy(self._next_states[indices].copy())
        dones = torch.from_numpy(self._dones[indices].copy())

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self._size

    def is_ready(self, min_samples: int) -> bool:
        return self._size >= min_samples

    def clear(self):
        self._write_idx = 0
        self._size = 0

    def save(self, path: str):
        """Save buffer contents to disk (single array dump — fast)."""
        data = {
            "capacity": self.capacity,
            "size": self._size,
            "write_idx": self._write_idx,
            "normalize_reward": self.normalize_reward,
            "reward_normalizer": self.reward_normalizer.state_dict() if self.reward_normalizer else None,
            "states": self._states[:self._size],
            "actions": self._actions[:self._size],
            "rewards": self._rewards[:self._size],
            "next_states": self._next_states[:self._size],
            "dones": self._dones[:self._size],
        }
        # np.savez_compressed always appends .npz, so strip it if present
        # so that the temp file ends up as <path>.
        base_path = path[:-4] if path.endswith(".npz") else path
        tmp_path = base_path + ".tmp"
        np.savez_compressed(tmp_path, **data)
        os.replace(tmp_path + ".npz", path)
        logger.info(f"ReplayBuffer saved: {self._size} transitions -> {path}")

    def load(self, path: str):
        """Load buffer contents from disk."""
        loaded = np.load(path, allow_pickle=True)
        self.capacity = int(loaded["capacity"])
        self._size = int(loaded["size"])
        self._write_idx = int(loaded["write_idx"])

        # Reallocate arrays with correct capacity
        state_shape = loaded["states"].shape[1:]
        self._states = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.float32)

        n = self._size
        self._states[:n] = loaded["states"]
        self._actions[:n] = loaded["actions"]
        self._rewards[:n] = loaded["rewards"]
        self._next_states[:n] = loaded["next_states"]
        self._dones[:n] = loaded["dones"]

        self.normalize_reward = bool(loaded.get("normalize_reward", self.normalize_reward))
        if self.normalize_reward and loaded.get("reward_normalizer"):
            if self.reward_normalizer is None:
                self.reward_normalizer = RunningRewardNormalizer()
            rn = loaded["reward_normalizer"].item() if loaded["reward_normalizer"].ndim == 0 else loaded["reward_normalizer"]
            self.reward_normalizer.load_state_dict(dict(rn))
        logger.info(f"ReplayBuffer loaded: {self._size} transitions from {path}")


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
                 normalize_reward: bool = True,
                 priority_mode: str = "proportional"):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling correction
            beta_end: Final beta value (annealed over beta_anneal_steps)
            beta_anneal_steps: Number of sampling steps to anneal beta
            epsilon: Small constant added to priorities to avoid zero probability
            normalize_reward: Whether to normalize rewards with running stats
            priority_mode: "proportional" (original) or "rank" (rank-based PER).
                Rank-based assigns priority = 1/(rank^alpha) based on TD-error
                ordering, making it more robust to outlier TD-error magnitudes.
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
        self.priority_mode = priority_mode

        if priority_mode == "rank":
            # Precompute normalizing constant for rank-based IS weights:
            # sum_{k=1}^{N} 1/k^alpha
            self._rank_norm = sum(1.0 / (k ** alpha) for k in range(1, capacity + 1))

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
        """Update priorities based on TD errors.

        Proportional mode: priority = (|delta| + epsilon)^alpha
        Rank-based mode: re-sort all transitions by |delta|, assign
            priority = 1 / (rank^alpha), then update the entire tree.
        """
        if self.priority_mode == "rank":
            self._update_rank_priorities(indices, td_errors)
        else:
            for idx, td_error in zip(indices, td_errors):
                priority = (abs(td_error) + self.epsilon) ** self.alpha
                self.tree.update(idx, priority)
                self.max_priority = max(self.max_priority, abs(td_error) + self.epsilon)

    def _update_rank_priorities(self, sampled_indices: np.ndarray,
                                td_errors: np.ndarray):
        """Rank-based: re-rank all N transitions and update tree.

        Stores the TD error on each Transition object so that subsequent
        re-ranking can access it without modifying the SumTree structure.
        """
        n = self.tree.size
        # 1) Collect all (abs_td_error, data_idx) pairs
        errors = np.zeros(n, dtype=np.float64)
        for i in range(n):
            t = self.tree.data[i]
            if t is not None and hasattr(t, '_last_td_error'):
                errors[i] = abs(t._last_td_error)

        # 2) Patch in fresh TD errors for sampled indices
        for j, (idx, td) in enumerate(zip(sampled_indices, td_errors)):
            data_idx = idx - self.capacity + 1
            if 0 <= data_idx < n and self.tree.data[data_idx] is not None:
                self.tree.data[data_idx]._last_td_error = float(abs(td))
                errors[data_idx] = float(abs(td))

        # 3) Sort by error (ascending) to get ranks; ties broken arbitrarily
        sorted_order = np.argsort(errors, kind='stable')
        ranks = np.empty(n, dtype=np.float64)
        ranks[sorted_order] = np.arange(1, n + 1, dtype=np.float64)

        # 4) Assign priorities: 1 / (rank^alpha), then normalize to tree range
        rank_priorities = 1.0 / (ranks ** self.alpha)
        rank_priorities /= self._rank_norm  # normalize so total weight is correct

        # 5) Batch-update tree leaf nodes
        for i in range(n):
            tree_idx = i + self.capacity - 1
            self.tree.update(tree_idx, float(rank_priorities[i]))

    def __len__(self) -> int:
        return self.tree.size

    def is_ready(self, min_samples: int) -> bool:
        return self.tree.size >= min_samples

    def clear(self):
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self.sample_count = 0

    def save(self, path: str):
        """Save PER buffer to disk."""
        data = {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_anneal_steps": self.beta_anneal_steps,
            "epsilon": self.epsilon,
            "max_priority": self.max_priority,
            "sample_count": self.sample_count,
            "normalize_reward": self.normalize_reward,
            "priority_mode": self.priority_mode,
            "reward_normalizer": self.reward_normalizer.state_dict() if self.reward_normalizer else None,
            "transitions": [],
            "priorities": [],
        }
        for i in range(self.tree.size):
            t = self.tree.data[i]
            if t is not None:
                data["transitions"].append({
                    "state": t.state,
                    "action": t.action,
                    "reward": t.reward,
                    "next_state": t.next_state,
                    "done": t.done,
                })
                data["priorities"].append(float(self.tree.tree[self.tree.capacity - 1 + i]))

        # np.savez_compressed always appends .npz, so strip it if present
        # so that the temp file ends up as <path>.
        base_path = path[:-4] if path.endswith(".npz") else path
        tmp_path = base_path + ".tmp"
        np.savez_compressed(tmp_path, data=np.array([data], dtype=object))
        os.replace(tmp_path + ".npz", path)
        logger.info(f"PrioritizedReplayBuffer saved: {self.tree.size} transitions -> {path}")

    def load(self, path: str):
        """Load PER buffer from disk."""
        loaded = np.load(path, allow_pickle=True)
        data = loaded["data"][0]
        self.capacity = data["capacity"]
        self.alpha = data.get("alpha", self.alpha)
        self.beta_start = data.get("beta_start", self.beta_start)
        self.beta_end = data.get("beta_end", self.beta_end)
        self.beta_anneal_steps = data.get("beta_anneal_steps", self.beta_anneal_steps)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.max_priority = data.get("max_priority", 1.0)
        self.sample_count = data.get("sample_count", 0)
        self.priority_mode = data.get("priority_mode", self.priority_mode)
        self.tree = SumTree(self.capacity)
        if self.priority_mode == "rank":
            self._rank_norm = sum(1.0 / (k ** self.alpha) for k in range(1, self.capacity + 1))
        self.normalize_reward = data.get("normalize_reward", self.normalize_reward)
        if self.normalize_reward and data.get("reward_normalizer"):
            if self.reward_normalizer is None:
                self.reward_normalizer = RunningRewardNormalizer()
            self.reward_normalizer.load_state_dict(data["reward_normalizer"])
        priorities = data.get("priorities", [])
        for i, t in enumerate(data["transitions"]):
            priority = priorities[i] if i < len(priorities) else self.max_priority ** self.alpha
            self.tree.add(priority, Transition(
                state=t["state"],
                action=t["action"],
                reward=float(t["reward"]),
                next_state=t["next_state"],
                done=t["done"],
            ))
        logger.info(f"PrioritizedReplayBuffer loaded: {self.tree.size} transitions (mode={self.priority_mode}) from {path}")
