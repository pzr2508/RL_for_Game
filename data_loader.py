import os
import cv2
import csv
import ast
import torch
import numpy as np
from utils.my_logger import logger
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from core.vision_engine import VisionEngine


def _load_frame(video_dir: str, frame_idx: int, img_size: int):
    """
    优先加载预处理的 .npy 文件，若不存在则回退到读取 .jpg 并实时处理。
    返回 (H, W) float32 数组或 None。
    """
    npy_path = os.path.join(video_dir, f"{frame_idx}.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path)

    # fallback: 从 jpg 读取并 resize
    img_path = os.path.join(video_dir, f"{frame_idx}.jpg")
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = VisionEngine._letterbox(img, img_size)
    return resized.astype(np.float32) / 255.0


@dataclass
class TrainSample:
    video_dir: str
    ctrl_a_frame: int
    action: int
    reward_ids: list
    reward_sum: int
    done: int


class TrainDataset(Dataset):
    def __init__(self, cache, img_size=640, continue_num=5, gap_num=1,
                 reward_mean: float = 0.0, reward_std: float = 1.0):
        """
        :param continue_num: 连续取多少帧作为一个state
        :param gap_num: s1与s2之间间隔 gap_num * 2 帧
        :param reward_mean: reward 均值（用于 normalization）
        :param reward_std: reward 标准差（用于 normalization）
        :return:
        """
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
        done_flag = s.done

        s1_list, s2_list, a_list, r_list, done_list = [], [], [], [], []

        for offset in [-1, 0, 1]:
            frame_idx = ctrl_a_frame + offset

            # 计算state (s1)
            s_frames = []
            s_end = frame_idx - self.gap_num
            s_start = s_end - self.continue_num

            for i in range(s_start, s_end):
                frame = _load_frame(video_dir, i, self.img_size)
                if frame is None:
                    break
                s_frames.append(frame)

            if len(s_frames) < self.continue_num:
                continue

            s1 = torch.from_numpy(np.stack(s_frames, axis=0))

            # 计算next state (s2)
            s2_frames = []
            s2_start = frame_idx + self.gap_num + 1
            s2_end = s2_start + self.continue_num

            for i in range(s2_start, s2_end):
                frame = _load_frame(video_dir, i, self.img_size)
                if frame is None:
                    break
                s2_frames.append(frame)

            if len(s2_frames) < self.continue_num:
                continue

            s2 = torch.from_numpy(np.stack(s2_frames, axis=0))

            s1_list.append(s1)
            s2_list.append(s2)
            a_list.append(torch.tensor(action, dtype=torch.int32))
            normalized_reward = (float(reward_sum) - self.reward_mean) / self.reward_std
            r_list.append(torch.tensor(normalized_reward, dtype=torch.float32))
            done_list.append(torch.tensor(done_flag, dtype=torch.float32))

        if len(s1_list) == 0 or len(s2_list)==0 or len(a_list)==0 or len(r_list)==0 or len(done_list)==0:
            return None, None, None, None, None

        # 补齐到3个样本
        while len(s1_list) < 3:
            s1_list.append(s1_list[-1])
            s2_list.append(s2_list[-1])
            a_list.append(a_list[-1])
            r_list.append(r_list[-1])
            done_list.append(done_list[-1])

        s1 = torch.stack(s1_list[:3])
        s2 = torch.stack(s2_list[:3])
        a = torch.stack(a_list[:3])
        r = torch.stack(r_list[:3])
        done = torch.stack(done_list[:3])
        return s1, s2, a, r, done


def load_csv_to_cache(csv_path):
    cache = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reward = ast.literal_eval(row["rewards"])
            actions = ast.literal_eval(row["actions"])
            for action in actions:
                sample = TrainSample(
                    video_dir=row["video_dir"],
                    ctrl_a_frame=int(row["ctrl_a_frame"]),
                    action=int(action),
                    reward_ids=reward,
                    reward_sum=int(row["reward_sum"]),
                    done=int(row["done"])
                )
                cache.append(sample)
    return cache


def compute_reward_stats(cache):
    """Compute mean and std of reward_sum across the training cache for normalization."""
    if not cache:
        return 0.0, 1.0
    rewards = np.array([float(s.reward_sum) for s in cache], dtype=np.float32)
    mean = float(rewards.mean())
    std = float(rewards.std())
    if std < 1e-8:
        std = 1.0
    logger.info(f"Reward normalization stats: mean={mean:.4f}, std={std:.4f}")
    return mean, std