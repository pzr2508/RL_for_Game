"""
Learned Reward Model for online RL.

The reward model is a neural network that predicts reward given (state, action).
It is first pretrained on offline training data (CSV), then continues to update
during online learning using a self-supervised temporal consistency objective.

Architecture:
  - Shares a similar CNN+Transformer frame encoder as DuelingDQN
  - Takes (state_frames, action_id) -> scalar reward prediction

Training:
  1. Pretrain: supervised on offline (state, action, reward) from CSV
  2. Online update: periodically fine-tune on a mix of:
     - Original offline data (replay from pretrain buffer)
     - TD-consistency signal: r_pred \approx Q(s,a) - \gamma * Q(s'\'',a'')
       (use the DQN'\''s Q-values as weak supervision)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from typing import Optional, Tuple

from utils.my_logger import logger
from rl.agent import SharedBackbone, AgentConfig


class RewardNet(nn.Module):
    """
    Reward prediction network built on top of DuelingDQN's full backbone.

    Architecture:
      DuelingDQN.forward_features()  ->  512-dim fused feature
      + action embedding             ->  reward head -> scalar

    This guarantees identical feature quality with the Q-network
    (ResidualBlock CNN + Transformer + attention pooling + fusion).
    """

    def __init__(self, input_frames: int, num_actions: int, backbone: SharedBackbone,
                 model_dim: int = 256):
        super().__init__()
        self.input_frames = input_frames
        self.num_actions = num_actions
        self.model_dim = model_dim

        # Shared backbone (same object as the DQN's online_net.backbone)
        self.backbone = backbone

        # Action embedding to 512-dim (matching backbone fusion output)
        self.action_embed = nn.Embedding(num_actions, 512)

        # Reward prediction head on [fused_feat(512), action_emb(512)]
        self.reward_head = nn.Sequential(
            nn.Linear(512 * 2, model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, frames, H, W) float32
            action: (batch,) int64 action indices

        Returns:
            reward: (batch,) predicted reward scalar
        """
        # Shared backbone: CNN + Transformer + attention + fusion
        fused_feat = self.backbone(state)  # (batch, 512)

        # Action embedding
        action_feat = self.action_embed(action)

        # Combine features + action -> reward
        combined = torch.cat([fused_feat, action_feat], dim=1)
        reward = self.reward_head(combined).squeeze(1)

        return reward


class RewardModel:
    """
    Learned reward model with pretrain + online update capabilities.

    Workflow:
        1. pretrain(offline_data) - Learn from CSV dataset
        2. predict(state, action) -> reward - Used during online play
        3. update_online(batch) - Fine-tune during online learning

    The online update uses two signals:
        a) Replay of original offline data (prevent catastrophic forgetting)
        b) TD-consistency: reward should be consistent with Q-value differences
           r_model(s,a) \approx Q(s,a) - \gamma * max_a'\'' Q(s'\'',a'\'')
    """

    def __init__(
        self,
        input_frames: int,
        num_actions: int,
        hidden_dim: int = 256,
        backbone: Optional[SharedBackbone] = None,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_dropout: float = 0.1,
        lr: float = 3e-4,
        device: Optional[str] = None,
        reward_mean: float = 0.0,
        reward_std: float = 1.0,
        gpu_ids: Optional[list] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        self.input_frames = input_frames
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.gpu_ids = gpu_ids or []

        # Resolve multi-GPU
        if torch.cuda.is_available() and self.gpu_ids:
            gpu_count = torch.cuda.device_count()
            self.gpu_ids = [gid for gid in self.gpu_ids if 0 <= gid < gpu_count]
            if self.gpu_ids:
                self.device = f"cuda:{self.gpu_ids[0]}"

        if backbone is None:
            # Create a standalone backbone with config-consistent architecture
            # (uses the same transformer parameters as the DQN config)
            bb_cfg = AgentConfig(
                model_dim=hidden_dim,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                transformer_dropout=transformer_dropout,
            )
            backbone = SharedBackbone(input_frames, bb_cfg)

        self.net = RewardNet(input_frames, num_actions, backbone, model_dim=hidden_dim).to(self.device)

        if len(self.gpu_ids) > 1:
            self.net = nn.DataParallel(self.net, device_ids=self.gpu_ids, output_device=self.gpu_ids[0])
            logger.info(f"RewardModel: DataParallel on GPUs {self.gpu_ids}")

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # Online update buffer: stores recent (state, action, td_target) for fine-tuning
        self.online_buffer = deque(maxlen=10000)
        # Offline replay buffer: stores a subset of pretrain data for anti-forgetting
        self.offline_replay = deque(maxlen=5000)

        self._update_steps = 0
        self._pretrain_loss = float("inf")

    def predict(self, state: np.ndarray, action: int) -> float:
        """
        Predict reward for a single (state, action) pair.

        Args:
            state: (frames, H, W) numpy array
            action: action index

        Returns:
            Predicted reward (denormalized to original scale)
        """
        self.net.eval()
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            a = torch.tensor([action], dtype=torch.long, device=self.device)
            r_norm = self.net(s, a).item()

        # Denormalize
        reward = r_norm * self.reward_std + self.reward_mean
        return reward

    def predict_batch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict rewards for a batch. Returns denormalized rewards."""
        self.net.eval()
        with torch.no_grad():
            s = states.to(self.device, dtype=torch.float32)
            a = actions.to(self.device, dtype=torch.long)
            r_norm = self.net(s, a)
        return r_norm * self.reward_std + self.reward_mean

    def pretrain_step(self, states: torch.Tensor, actions: torch.Tensor,
                      rewards: torch.Tensor) -> float:
        """
        One supervised training step on offline data.

        Args:
            states: (batch, frames, H, W)
            actions: (batch,) action indices
            rewards: (batch,) normalized reward targets

        Returns:
            loss value
        """
        self.net.train()
        s = states.to(self.device, dtype=torch.float32)
        a = actions.to(self.device, dtype=torch.long)
        r_target = rewards.to(self.device, dtype=torch.float32)

        r_pred = self.net(s, a)
        loss = F.smooth_l1_loss(r_pred, r_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        self._pretrain_loss = loss.item()

        # Store some samples for offline replay during online phase
        if len(self.offline_replay) < self.offline_replay.maxlen:
            batch_size = s.shape[0]
            indices = random.sample(range(batch_size), min(4, batch_size))
            for i in indices:
                self.offline_replay.append((
                    s[i].cpu().numpy(),
                    int(a[i].item()),
                    float(r_target[i].item()),
                ))

        return loss.item()

    def populate_offline_replay(self, cache, img_size: int, continue_num: int,
                                 gap_num: int = 1, max_samples: int = 200,
                                 reward_mean: Optional[float] = None,
                                 reward_std: Optional[float] = None):
        """
        Populate offline_replay from a cache of TrainSample objects.
        This provides anti-forgetting data during online learning.

        Args:
            cache: list of TrainSample from data_loader.load_csv_to_cache()
            img_size: frame resize target size
            continue_num: number of frames per state
            gap_num: frame gap before ctrl_a_frame
            max_samples: max number of samples to store
            reward_mean: normalization mean (uses self.reward_mean if None)
            reward_std: normalization std (uses self.reward_std if None)
        """
        from data_loader import _load_frame

        mean = reward_mean if reward_mean is not None else self.reward_mean
        std = reward_std if reward_std is not None else self.reward_std

        if not cache:
            logger.warning("populate_offline_replay: cache is empty, nothing to add.")
            return

        # Randomly sample a diverse subset (better than even-spaced when cache >> max_samples)
        sample_size = min(len(cache), max_samples * 2)  # draw more to account for build failures
        candidate_indices = random.sample(range(len(cache)), sample_size)

        added = 0
        for idx in candidate_indices:
            sample = cache[idx]
            video_dir = sample.video_dir
            ctrl_a_frame = sample.ctrl_a_frame

            # Build state frames
            s_end = ctrl_a_frame - gap_num
            s_start = s_end - continue_num
            s_frames = []
            for i in range(s_start, s_end):
                frame = _load_frame(video_dir, i, img_size)
                if frame is None:
                    break
                s_frames.append(frame)

            if len(s_frames) < continue_num:
                continue

            # Avoid duplicates (same video_dir + frame + action)
            dup_key = (sample.video_dir, sample.ctrl_a_frame, sample.action)
            if hasattr(self, '_offline_keys') and dup_key in self._offline_keys:
                continue
            if not hasattr(self, '_offline_keys'):
                self._offline_keys = set()
            self._offline_keys.add(dup_key)

            state = np.stack(s_frames, axis=0).astype(np.float32)
            reward_norm = (float(sample.reward_sum) - mean) / std

            self.offline_replay.append((
                state,
                int(sample.action),
                float(reward_norm),
            ))
            added += 1
            if added >= max_samples:
                break

        logger.info(
            f"populate_offline_replay: added {added} samples "
            f"(offline_replay size={len(self.offline_replay)}, "
            f"maxlen={self.offline_replay.maxlen})"
        )
        self._offline_keys = None  # release memory

    def refresh_offline_replay(self, cache, img_size: int, continue_num: int,
                                gap_num: int = 1, max_samples: int = 200,
                                keep_ratio: float = 0.5,
                                reward_mean: Optional[float] = None,
                                reward_std: Optional[float] = None):
        """
        Periodically refresh offline_replay to keep it diverse.
        Retains a fraction of existing samples and adds fresh random ones.

        Args:
            cache: list of TrainSample from data_loader.load_csv_to_cache()
            img_size: frame resize target size
            continue_num: number of frames per state
            gap_num: frame gap before ctrl_a_frame
            max_samples: max total samples after refresh
            keep_ratio: fraction of existing samples to retain (0=replace all, 1=keep all)
            reward_mean: normalization mean (uses self.reward_mean if None)
            reward_std: normalization std (uses self.reward_std if None)
        """
        if not cache:
            logger.warning("refresh_offline_replay: cache is empty, skip.")
            return

        # Keep a random subset of existing samples
        old_samples = list(self.offline_replay)
        keep_count = int(len(old_samples) * keep_ratio)
        kept = set(random.sample(range(len(old_samples)), min(keep_count, len(old_samples))))
        new_replay = deque(maxlen=self.offline_replay.maxlen)
        for i, s in enumerate(old_samples):
            if i in kept:
                new_replay.append(s)

        self.offline_replay = new_replay

        # Fill remaining slots with fresh random samples from cache
        remaining = max_samples - len(self.offline_replay)
        if remaining > 0:
            self.populate_offline_replay(
                cache, img_size, continue_num,
                gap_num=gap_num, max_samples=remaining,
                reward_mean=reward_mean, reward_std=reward_std,
            )

        logger.info(
            f"refresh_offline_replay done: {len(self.offline_replay)} samples "
            f"(kept {keep_count}, added fresh up to {remaining})"
        )

    def update_online(
        self,
        state: np.ndarray,
        action: int,
        q_agent,
        next_state: np.ndarray,
        gamma: float = 0.99,
        offline_mix_ratio: float = 0.5,
        min_batch: int = 16,
        update_every: int = 8,
        num_train_steps: int = 5,
    ) -> Optional[float]:
        """
        Online update using TD-consistency + offline replay.

        The TD-consistency signal:
            r_target = Q(s,a) - \gamma * max_a'\'' Q(s'\'', a'\'')

        Args:
            state: current state (frames, H, W)
            action: taken action
            q_agent: RLAgent instance (to compute Q-values)
            next_state: resulting next state
            gamma: discount factor
            offline_mix_ratio: fraction of batch from offline replay
            min_batch: minimum batch size to trigger update
            update_every: update reward model every N calls
            num_train_steps: number of gradient steps per effective update

        Returns:
            loss if update happened, None otherwise
        """
        # Compute TD-consistency target from Q-network
        with torch.no_grad():
            s_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            s_next_t = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)

            q_sa = q_agent.online_net(s_t)[0, action].item()
            q_next_max = q_agent.target_net(s_next_t).max(dim=1)[0].item()
            td_target = q_sa - gamma * q_next_max

        # Normalize the TD target
        td_target_norm = (td_target - self.reward_mean) / self.reward_std

        # Store in online buffer
        self.online_buffer.append((state, action, td_target_norm))

        # Check if we have enough samples
        if len(self.online_buffer) < min_batch:
            return None

        # Only update periodically
        self._update_steps += 1
        if self._update_steps % update_every != 0:
            return None

        # Perform multiple gradient steps per update for better convergence
        total_loss = 0.0
        for _ in range(num_train_steps):
            # Build mixed batch: online + offline replay (re-sampled each step)
            n_offline = int(min_batch * offline_mix_ratio)
            n_online = min_batch - n_offline

            batch_states = []
            batch_actions = []
            batch_rewards = []

            # Sample from online buffer
            sample_size = min(n_online, len(self.online_buffer))
            if sample_size < 1:
                break
            online_samples = random.sample(self.online_buffer, sample_size)
            for s, a, r in online_samples:
                batch_states.append(s)
                batch_actions.append(a)
                batch_rewards.append(r)

            # Sample from offline replay (anti-forgetting)
            if self.offline_replay and n_offline > 0:
                offline_samples = random.sample(
                    self.offline_replay,
                    min(n_offline, len(self.offline_replay))
                )
                for s, a, r in offline_samples:
                    batch_states.append(s)
                    batch_actions.append(a)
                    batch_rewards.append(r)

            if len(batch_states) < 2:
                break

            # Train step
            self.net.train()
            s_batch = torch.from_numpy(np.stack(batch_states)).float().to(self.device)
            a_batch = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
            r_batch = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)

            r_pred = self.net(s_batch, a_batch)
            step_loss = F.smooth_l1_loss(r_pred, r_batch)

            self.optimizer.zero_grad()
            step_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            total_loss += float(step_loss.item())

        # Enrich offline_replay with high-error online experiences (from last step)
        if len(batch_states) >= 2:
            with torch.no_grad():
                per_sample_loss = (r_pred - r_batch).abs()
            best_idx = int(per_sample_loss.argmax().item())
            if best_idx < len(online_samples):
                best_s, best_a, best_r = online_samples[best_idx]
                if len(self.offline_replay) >= self.offline_replay.maxlen:
                    # Replace a random old sample to keep size bounded
                    replace_idx = random.randrange(len(self.offline_replay))
                    old_list = list(self.offline_replay)
                    old_list[replace_idx] = (best_s, best_a, best_r)
                    self.offline_replay = deque(old_list, maxlen=self.offline_replay.maxlen)
                else:
                    self.offline_replay.append((best_s, best_a, best_r))

        avg_loss = total_loss / max(1, num_train_steps)
        return avg_loss

    def _get_net_module(self):
        """Get the underlying module (unwrap DataParallel if needed)."""
        if isinstance(self.net, nn.DataParallel):
            return self.net.module
        return self.net

    def save(self, path: str):
        """Save reward model checkpoint."""
        net_to_save = self._get_net_module()
        # Persist a limited snapshot of offline_replay and online_buffer
        # (full state arrays are large, so only keep the most recent entries)
        offline_snapshot = list(self.offline_replay)[-100:] if self.offline_replay else []
        online_snapshot = list(self.online_buffer)[-50:] if self.online_buffer else []
        torch.save({
            "net": net_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "update_steps": self._update_steps,
            "offline_replay": offline_snapshot,
            "online_buffer": online_snapshot,
        }, path)
        logger.info(f"RewardModel saved to {path}")

    def load(self, path: str):
        """Load reward model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        target = self._get_net_module()
        target.load_state_dict(checkpoint["net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.reward_mean = checkpoint.get("reward_mean", 0.0)
        self.reward_std = checkpoint.get("reward_std", 1.0)
        self._update_steps = checkpoint.get("update_steps", 0)
        # Restore offline replay and online buffer snapshots
        if "offline_replay" in checkpoint and checkpoint["offline_replay"]:
            self.offline_replay.clear()
            for sample in checkpoint["offline_replay"]:
                self.offline_replay.append(tuple(sample))
            logger.info(f"Restored offline_replay: {len(self.offline_replay)} samples")
        if "online_buffer" in checkpoint and checkpoint["online_buffer"]:
            self.online_buffer.clear()
            for sample in checkpoint["online_buffer"]:
                self.online_buffer.append(tuple(sample))
            logger.info(f"Restored online_buffer: {len(self.online_buffer)} samples")
        logger.info(f"RewardModel loaded from {path} (update_steps={self._update_steps})")
