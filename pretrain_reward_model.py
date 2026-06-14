"""
Pretrain the reward model from offline CSV data.

Usage:
    python pretrain_reward_model.py [--config config/config.yaml] [--epochs 100]

This trains a reward model that maps (state, action) -> reward,
using the same offline data as DQN training. The trained model
is then used during online learning to provide reward signals.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import load_csv_to_cache, compute_reward_stats, _load_frame
from rl.reward_model import RewardModel
from utils.config_loader import load_config
from utils.my_logger import logger


class RewardDataset(Dataset):
    """Dataset for reward model pretraining: (state, action) -> reward."""

    def __init__(self, cache, img_size: int = 640, continue_num: int = 10,
                 gap_num: int = 1, reward_mean: float = 0.0, reward_std: float = 1.0):
        self.cache = cache
        self.img_size = img_size
        self.continue_num = continue_num
        self.gap_num = gap_num
        self.reward_mean = reward_mean
        self.reward_std = reward_std

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        s = self.cache[idx]
        video_dir = s.video_dir
        ctrl_a_frame = s.ctrl_a_frame
        action = s.action
        reward_sum = s.reward_sum

        # Build state (same logic as TrainDataset)
        s_end = ctrl_a_frame - self.gap_num
        s_start = s_end - self.continue_num

        s_frames = []
        for i in range(s_start, s_end):
            frame = _load_frame(video_dir, i, self.img_size)
            if frame is None:
                break
            s_frames.append(frame)

        if len(s_frames) < self.continue_num:
            # Return zeros as placeholder (will be filtered by collate)
            return None, None, None

        state = torch.from_numpy(np.stack(s_frames, axis=0))  # (frames, H, W)
        action_t = torch.tensor(action, dtype=torch.long)
        reward_norm = (float(reward_sum) - self.reward_mean) / self.reward_std
        reward_t = torch.tensor(reward_norm, dtype=torch.float32)

        return state, action_t, reward_t


def reward_collate_fn(batch):
    """Filter out None samples and stack."""
    valid = [(s, a, r) for s, a, r in batch if s is not None]
    if not valid:
        return None, None, None
    states, actions, rewards = zip(*valid)
    return torch.stack(states), torch.stack(actions), torch.stack(rewards)


def _parse_gpu_ids(raw):
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def pretrain_reward_model(config_path: str, epochs: int = 100, batch_size: int = 32,
                          lr: float = 3e-4, save_path: str = "models/reward_model.pth",
                          gpu_ids: list = None):
    config = load_config(config_path)
    ai_cfg = config.get("ai", {})
    path_cfg = config.get("paths", {})
    train_cfg = config.get("train", {})
    rl_cfg = config.get("rl", {})
    online_cfg = config.get("online", {})

    state_frames = ai_cfg.get("continue_frames_num", 10)
    frame_size = ai_cfg.get("frams_resize", 640)
    records_csv = path_cfg.get("records_csv", "train_data/records.csv")
    num_actions = 26

    # Resolve GPU ids: CLI > config
    if gpu_ids is None:
        gpu_ids = _parse_gpu_ids(train_cfg.get("gpu_ids", []))
    if gpu_ids:
        logger.info(f"Reward model training on GPUs: {gpu_ids}")

    # Load offline data
    logger.info(f"Loading training data from: {records_csv}")
    cache = load_csv_to_cache(records_csv)
    reward_mean, reward_std = compute_reward_stats(cache)
    logger.info(f"Total samples: {len(cache)}, reward_mean={reward_mean:.4f}, reward_std={reward_std:.4f}")

    # Create dataset & loader
    dataset = RewardDataset(
        cache, img_size=frame_size, continue_num=state_frames,
        gap_num=1, reward_mean=reward_mean, reward_std=reward_std,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True, collate_fn=reward_collate_fn,
    )

    # Create reward model
    reward_model = RewardModel(
        input_frames=state_frames,
        num_actions=num_actions,
        hidden_dim=online_cfg.get("reward_model", {}).get("hidden_dim", 256),
        transformer_layers=rl_cfg.get("transformer_layers", 2),
        transformer_heads=rl_cfg.get("transformer_heads", 8),
        transformer_dropout=rl_cfg.get("transformer_dropout", 0.1),
        lr=lr,
        reward_mean=reward_mean,
        reward_std=reward_std,
        gpu_ids=gpu_ids,
    )

    writer = SummaryWriter("runs/reward_model")
    global_step = 0

    logger.info(f"Starting reward model pretraining: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        step_count = 0

        for states, actions, rewards in loader:
            if states is None:
                continue

            loss = reward_model.pretrain_step(states, actions, rewards)
            epoch_loss += loss
            step_count += 1
            global_step += 1

            if global_step % 100 == 0:
                writer.add_scalar("reward_model/pretrain_loss", loss, global_step)

        if step_count > 0:
            avg_loss = epoch_loss / step_count
            logger.info(f"Epoch {epoch}/{epochs} - avg_loss={avg_loss:.6f}, steps={step_count}")
            writer.add_scalar("reward_model/epoch_loss", avg_loss, epoch)

        # Save periodically
        if epoch % 20 == 0 or epoch == epochs:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            reward_model.save(save_path)

    writer.close()
    logger.info(f"Reward model pretraining complete. Saved to: {save_path}")
    return reward_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain reward model from offline data")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-path", default="models/reward_model.pth")
    parser.add_argument("--gpus", default=None, help="GPU ids, e.g. '0,1,2,3'")
    args = parser.parse_args()

    pretrain_reward_model(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        gpu_ids=_parse_gpu_ids(args.gpus) if args.gpus else None,
    )
