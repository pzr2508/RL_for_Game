import os
import time
import torch
import argparse
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from core.frame_buffer import FrameBuffer
from core.recorder import ScreenRecorder
from core.vision_engine import VisionEngine
from data_loader import TrainDataset, load_csv_to_cache, compute_reward_stats

from rl.agent import RLAgent
from rl.trainer import Trainer
from utils.config_loader import load_config
from utils.my_logger import logger


ACTION_NUM = 26


def _log_cache_action_stats(cache, action_num: int):
    if not cache:
        logger.warning("Training cache is empty; skip action distribution logging.")
        return

    count_map = defaultdict(int)
    reward_sum_map = defaultdict(float)

    for sample in cache:
        action_id = int(sample.action)
        count_map[action_id] += 1
        reward_sum_map[action_id] += float(sample.reward_sum)

    total = sum(count_map.values())
    logger.info(f"Training label summary: total_samples={total}, action_space={action_num}")

    for action_id in range(action_num):
        count = count_map.get(action_id, 0)
        if count == 0:
            continue
        ratio = (count / total) * 100.0
        avg_reward = reward_sum_map[action_id] / count
        logger.info(
            f"action={action_id:02d} count={count} ratio={ratio:.2f}% avg_reward={avg_reward:.2f}"
        )


def _build_action_balanced_sampler(cache, action_num: int, alpha: float = 1.0):
    if not cache:
        return None

    alpha = max(0.0, float(alpha))
    action_count = [0 for _ in range(action_num)]
    for sample in cache:
        action_id = int(sample.action)
        if 0 <= action_id < action_num:
            action_count[action_id] += 1

    weights = []
    for sample in cache:
        action_id = int(sample.action)
        if 0 <= action_id < action_num and action_count[action_id] > 0:
            base = 1.0 / float(action_count[action_id])
            weight = base ** alpha
        else:
            weight = 0.0
        weights.append(weight)

    if not any(w > 0 for w in weights):
        logger.warning("Action-balanced sampler has no valid positive weights; fallback to shuffle=True.")
        return None

    logger.info(f"Enable action-balanced sampling: alpha={alpha:.2f}")
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _parse_gpu_ids(raw_gpu_ids):
    if raw_gpu_ids is None:
        return None
    if isinstance(raw_gpu_ids, (list, tuple)):
        return [int(x) for x in raw_gpu_ids]

    text = str(raw_gpu_ids).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _resolve_auto_batch(cli_args, train_cfg):
    if cli_args.auto_batch is True:
        return True
    if cli_args.auto_batch is False:
        return False
    return bool(train_cfg.get("auto_batch_size", False))


def _auto_scale_batch_size(runtime):
    base_batch_size = int(runtime["batch_size"])
    if base_batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if not runtime["auto_batch_size"]:
        return base_batch_size

    if not torch.cuda.is_available():
        logger.warning("auto_batch_size enabled but CUDA is unavailable, fallback to configured batch_size.")
        return base_batch_size

    gpu_ids = runtime["train_gpu_ids"] if runtime["train_gpu_ids"] else [0]
    gpu_count = torch.cuda.device_count()
    valid_gpu_ids = [gid for gid in gpu_ids if 0 <= gid < gpu_count]
    if not valid_gpu_ids:
        logger.warning("auto_batch_size enabled but no valid GPU ids found, fallback to configured batch_size.")
        return base_batch_size

    gpu_mem_gb = [torch.cuda.get_device_properties(gid).total_memory / (1024 ** 3) for gid in valid_gpu_ids]
    min_mem_gb = min(gpu_mem_gb)
    ref_mem_gb = max(1.0, float(runtime["batch_ref_gpu_mem_gb"]))

    memory_scale = min_mem_gb / ref_mem_gb
    scaled_batch = max(1, int(round(base_batch_size * len(valid_gpu_ids) * memory_scale)))

    max_batch_size = runtime["max_batch_size"]
    if max_batch_size is not None and max_batch_size > 0:
        scaled_batch = min(scaled_batch, int(max_batch_size))

    logger.info(
        f"Auto batch scaling: base={base_batch_size}, gpus={valid_gpu_ids}, min_mem={min_mem_gb:.2f}GB, ref_mem={ref_mem_gb:.2f}GB, scaled={scaled_batch}")
    return scaled_batch


def flatten_group_collate(batch):
    valid = [item for item in batch if item[0] is not None]
    if not valid:
        return None, None, None, None, None

    s1, s2, a, r, done = zip(*valid)
    s1 = torch.stack(s1)
    s2 = torch.stack(s2)
    a = torch.stack(a)
    r = torch.stack(r)
    done = torch.stack(done)

    batch_size, group_num = s1.shape[:2]
    s1 = s1.view(batch_size * group_num, *s1.shape[2:])
    s2 = s2.view(batch_size * group_num, *s2.shape[2:])
    a = a.view(batch_size * group_num, *a.shape[2:])
    r = r.view(batch_size * group_num, *r.shape[2:])
    done = done.view(batch_size * group_num, *done.shape[2:])
    return s1, s2, a, r, done


def add_epoch_suffix(model_path: str, epoch: int) -> str:
    if model_path.endswith(".pth"):
        return model_path[:-4] + f"_{epoch}.pth"
    return f"{model_path}_{epoch}"


def _resolve_runtime_config(config: dict, cli_args):
    app_cfg = config.get("app", {})
    ai_cfg = config.get("ai", {})
    train_cfg = config.get("train", {})
    infer_cfg = config.get("inference", {})
    path_cfg = config.get("paths", {})
    rl_cfg = config.get("rl", {})

    runtime = {
        "mode": cli_args.mode,
        "train_fps": app_cfg.get("train_fps", 5),
        "recorder_fps": app_cfg.get("recorder_fps", 60),
        "state_frames": ai_cfg.get("continue_frames_num", 10),
        "frame_size": ai_cfg.get("frams_resize", 224),
        "model_path": cli_args.model_path or path_cfg.get("model_path", "models/dueling_dqn.pth"),
        "records_csv": cli_args.data_csv or path_cfg.get("records_csv", "train_data/records.csv"),
        "log_dir": cli_args.log_dir or path_cfg.get("tensorboard_dir", "runs/dueling_dqn"),
        "batch_size": cli_args.batch_size or train_cfg.get("batch_size", 8),
        "auto_batch_size": _resolve_auto_batch(cli_args, train_cfg),
        "batch_ref_gpu_mem_gb": cli_args.ref_gpu_mem_gb if cli_args.ref_gpu_mem_gb is not None else train_cfg.get("batch_ref_gpu_mem_gb", 8.0),
        "max_batch_size": cli_args.max_batch_size if cli_args.max_batch_size is not None else train_cfg.get("max_batch_size", None),
        "epochs": cli_args.epochs or train_cfg.get("epochs", 200),
        "start_epoch": cli_args.start_epoch if cli_args.start_epoch is not None else train_cfg.get("start_epoch", 1),
        "save_every": train_cfg.get("save_every", 20),
        "num_workers": train_cfg.get("num_workers", 2),
        "pin_memory": train_cfg.get("pin_memory", True),
        "balance_actions": train_cfg.get("balance_actions", False),
        "balance_alpha": train_cfg.get("balance_alpha", 1.0),
        "target_update": rl_cfg.get("target_update", 1000),
        "train_gpu_ids": _parse_gpu_ids(cli_args.gpus) if cli_args.gpus is not None else _parse_gpu_ids(train_cfg.get("gpu_ids", [])),
        "rl_agent_cfg": rl_cfg,
        "record_enable": config.get("record", {}).get("enable", True),
        "record_output_dir": config.get("record", {}).get("output_dir", "./videos"),
        "inference_sleep_ms": infer_cfg.get("sleep_ms_when_empty", 200),
        "online": config.get("online", {}),
    }

    return runtime


def run_train(runtime):
    model_dir = os.path.dirname(runtime["model_path"])
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    agent_cfg = dict(runtime["rl_agent_cfg"])
    agent_cfg["gpu_ids"] = runtime["train_gpu_ids"]

    agent = RLAgent(
        num_actions=ACTION_NUM,
        model_input_dime=runtime["state_frames"],
        target_update=runtime["target_update"],
        config=agent_cfg,
    )

    if runtime["train_gpu_ids"]:
        logger.info(f"Train GPU ids: {runtime['train_gpu_ids']}")
    else:
        logger.info("Train GPU ids: default(auto)")
    if os.path.exists(runtime["model_path"]):
        logger.info(f"Load model from: {runtime['model_path']}")
        agent.load(runtime["model_path"], strict=False)

    effective_batch_size = _auto_scale_batch_size(runtime)

    cache = load_csv_to_cache(runtime["records_csv"])
    _log_cache_action_stats(cache, ACTION_NUM)
    reward_mean, reward_std = compute_reward_stats(cache)

    sampler = None
    shuffle = True
    if runtime["balance_actions"]:
        sampler = _build_action_balanced_sampler(cache, ACTION_NUM, alpha=runtime["balance_alpha"])
        shuffle = sampler is None

    dataset = TrainDataset(
        cache,
        img_size=runtime["frame_size"],
        continue_num=runtime["state_frames"],
        gap_num=1,
        reward_mean=reward_mean,
        reward_std=reward_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=runtime["num_workers"],
        pin_memory=runtime["pin_memory"],
        collate_fn=flatten_group_collate,
    )

    trainer = Trainer(agent)
    writer = SummaryWriter(runtime["log_dir"])

    agent.set_train_mode()

    # Compute total training steps for LR scheduler cosine decay schedule
    total_train_steps = 0
    for _ in range(runtime["start_epoch"], runtime["start_epoch"] + runtime["epochs"]):
        total_train_steps += max(1, len(cache) // effective_batch_size)
    agent.scheduler = agent._build_lr_scheduler(total_train_steps)

    logger.info(f"Start training: epochs={runtime['epochs']}, batch_size={effective_batch_size}")
    for epoch in range(runtime["start_epoch"], runtime["start_epoch"] + runtime["epochs"]):
        epoch_loss = 0.0
        epoch_q = 0.0
        step_count = 0

        for s1, s2, a, r, done in loader:
            if s1 is None:
                continue

            loss, q_val = trainer.step_only_train(s1, a, r, s2, done)
            epoch_loss += loss
            epoch_q += q_val
            step_count += 1

        if step_count > 0:
            avg_loss = epoch_loss / step_count
            avg_q = epoch_q / step_count
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("train/q_mean", avg_q, epoch)
            logger.info(f"epoch={epoch} avg_loss={avg_loss:.6f} avg_q={avg_q:.4f} steps={step_count}")

        if epoch % runtime["target_update"] == 0:
            agent.sync_target_network()

        if epoch % runtime["save_every"] == 0:
            epoch_model = add_epoch_suffix(runtime["model_path"], epoch)
            agent.save(epoch_model)
            logger.info(f"Checkpoint saved: {epoch_model}")

    agent.save(runtime["model_path"])
    writer.close()
    logger.info(f"Training done. Final model saved: {runtime['model_path']}")


def run_inference(runtime, config):
    from core.bot_controller import BotController
    from core.screen_capture import ScreenCaptureThread
    from input.keyboard_controller import KeyboardController
    from input.mouse_controller import MouseControllerGame
    from logic.decision import DecisionEngine

    if not os.path.exists(runtime["model_path"]):
        raise FileNotFoundError(f"Model not found: {runtime['model_path']}")

    agent = RLAgent(
        num_actions=ACTION_NUM,
        model_input_dime=runtime["state_frames"],
        target_update=runtime["target_update"],
        config=runtime["rl_agent_cfg"],
    )
    agent.load(runtime["model_path"], strict=True)
    agent.set_eval_mode()

    trainer = Trainer(agent)
    keyboard = KeyboardController()
    mouse = MouseControllerGame()
    vision = VisionEngine(enable=True, history_len=runtime["state_frames"], out_size=runtime["frame_size"])
    decision = DecisionEngine(keyboard, mouse, agent)
    bot = BotController(vision, decision, trainer)
    bot.start()

    buffer = FrameBuffer()
    capture_thread = ScreenCaptureThread(buffer, config)
    capture_thread.start()

    recorder = ScreenRecorder(
        runtime["record_enable"],
        runtime["record_output_dir"],
        runtime["recorder_fps"],
        (config["screen"]["width"], config["screen"]["height"]),
    )

    frame_interval = max(1, int(runtime["recorder_fps"] / max(runtime["train_fps"], 1)))
    logger.info("Inference started. Press F5 to enable agent, F6 to disable agent.")

    frame_count = 0
    try:
        while True:
            frame = buffer.get()
            if frame is None:
                time.sleep(runtime["inference_sleep_ms"] / 1000.0)
                continue

            if frame_count % frame_interval == 0:
                bot.handle_frame_for_inference(frame)

            recorder.write(frame)
            frame_count += 1
            time.sleep(0.1)
    finally:
        capture_thread.stop()
        recorder.release()


def build_parser():
    parser = argparse.ArgumentParser(description="RL for game with Dueling DQN")
    parser.add_argument("--mode", choices=["train", "inference", "online", "preprocess"], default="inference")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-csv", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--auto-batch", dest="auto_batch", action="store_true", help="Enable auto batch size scaling by GPU memory")
    parser.add_argument("--no-auto-batch", dest="auto_batch", action="store_false", help="Disable auto batch size scaling")
    parser.add_argument("--ref-gpu-mem-gb", type=float, default=None, help="Reference GPU memory (GB) for base batch size")
    parser.add_argument("--max-batch-size", type=int, default=None, help="Upper bound for auto-scaled batch size")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--start-epoch", type=int, default=None)
    parser.add_argument("--gpus", default=None, help="Training GPU ids, e.g. '0,1,2'. Empty means auto/default device.")
    parser.add_argument("--preprocess-workers", type=int, default=8, help="Number of parallel workers for frame preprocessing")
    parser.add_argument("--preprocess-force", action="store_true", default=False,
                        help="Force reprocess all frames, overwrite existing .npy (default: incremental)")
    parser.set_defaults(auto_batch=None)
    return parser


def run_preprocess(runtime, args):
    """预处理所有训练帧为 .npy 文件，消除训练时的 I/O 瓶颈。"""
    from preprocess_frames import preprocess_all_videos
    import csv

    # 从 records.csv 中收集所有 video_dir
    csv_path = runtime["records_csv"]
    cache = load_csv_to_cache(csv_path)
    _log_cache_action_stats(cache, ACTION_NUM)

    video_dirs = set()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_dirs.add(os.path.normpath(row["video_dir"]))

    img_size = runtime["frame_size"]
    workers = args.preprocess_workers

    force = args.preprocess_force
    mode_str = "FORCE (overwrite all)" if force else "incremental (skip existing)"
    logger.info(f"Preprocessing {len(video_dirs)} video directories, img_size={img_size}, workers={workers}, mode={mode_str}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from preprocess_frames import preprocess_single_video

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

    logger.info(f"Preprocess done. processed={total_processed}, skipped={total_skipped}")


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)
    runtime = _resolve_runtime_config(config, args)

    if runtime["mode"] == "preprocess":
        run_preprocess(runtime, args)
    elif runtime["mode"] == "train":
        run_train(runtime)
    elif runtime["mode"] == "online":
        from online_train import run_online
        run_online(runtime, config)
    else:
        run_inference(runtime, config)


if __name__ == "__main__":
    main()
