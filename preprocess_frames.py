"""
预处理帧数据：将所有 .jpg 帧读取、灰度化、letterbox resize、归一化后保存为 .npy 文件。
训练时直接加载 .npy 文件，避免每次 __getitem__ 都从磁盘读取并 resize 图片的 I/O 瓶颈。

用法:
    python preprocess_frames.py --data_dir ./train_data/saved_videos --img_size 640 --workers 8
"""

import os
import cv2
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from core.vision_engine import VisionEngine
from utils.my_logger import logger


def preprocess_single_video(video_dir: str, img_size: int, force: bool = False) -> tuple:
    """
    预处理单个视频目录下的所有 .jpg 帧。
    :param force: 若为 True 则强制重新处理所有帧（覆盖已有 .npy）
    返回 (video_dir, processed_count, skipped_count)
    """
    processed = 0
    skipped = 0

    if not os.path.isdir(video_dir):
        return video_dir, 0, 0

    for fname in os.listdir(video_dir):
        if not fname.endswith(".jpg"):
            continue

        base_name = fname[:-4]  # 去掉 .jpg
        npy_path = os.path.join(video_dir, f"{base_name}.npy")

        # 如果 .npy 已存在且非强制模式则跳过
        if not force and os.path.exists(npy_path):
            skipped += 1
            continue

        img_path = os.path.join(video_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # letterbox resize
        resized = VisionEngine._letterbox(img, img_size)
        # 归一化为 float32
        state = resized.astype(np.float32) / 255.0

        # 保存为 .npy
        np.save(npy_path, state)
        processed += 1

    return video_dir, processed, skipped


def preprocess_all_videos(data_dir: str, img_size: int, workers: int = 8, force: bool = False):
    """
    遍历 data_dir 下所有子目录，并行预处理帧。
    :param force: 若为 True 则强制重新处理所有帧（覆盖已有 .npy）
    """
    # 收集所有视频子目录
    video_dirs = []
    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)
        if os.path.isdir(full_path):
            video_dirs.append(full_path)

    if not video_dirs:
        logger.warning(f"No video directories found in {data_dir}")
        return

    mode_str = "FORCE (overwrite all)" if force else "incremental (skip existing)"
    logger.info(f"Found {len(video_dirs)} video directories. Mode: {mode_str}")
    logger.info(f"Starting preprocessing with {workers} workers, target size: {img_size}x{img_size}")

    total_processed = 0
    total_skipped = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(preprocess_single_video, vdir, img_size, force): vdir
            for vdir in video_dirs
        }

        for future in as_completed(futures):
            vdir, processed, skipped = future.result()
            total_processed += processed
            total_skipped += skipped
            if processed > 0:
                dir_name = os.path.basename(vdir)
                logger.info(f"  [{dir_name}] processed={processed}, skipped(already exist)={skipped}")

    logger.info(f"Preprocessing done. total_processed={total_processed}, total_skipped={total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess video frames to .npy")
    parser.add_argument("--data_dir", type=str, default="./train_data/saved_videos",
                        help="Root directory containing video subdirectories")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Force reprocess all frames (overwrite existing .npy files)")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Target letterbox size (default: 640)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    preprocess_all_videos(args.data_dir, args.img_size, args.workers, force=args.force)
