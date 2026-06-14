import yaml
import os
from utils.my_logger import logger


DEFAULT_CONFIG = {
    "app": {"recorder_fps": 60, "train_fps": 5},
    "screen": {"monitor_id": 1, "width": 1920, "height": 1080},
    "ai": {"continue_frames_num": 10, "frams_resize": 224, "enable": True},
    "paths": {
        "model_path": "models/dueling_dqn.pth",
        "records_csv": "train_data/records.csv",
        "tensorboard_dir": "runs/dueling_dqn",
    },
    "train": {
        "batch_size": 8,
        "epochs": 200,
        "start_epoch": 1,
        "save_every": 20,
        "num_workers": 2,
        "pin_memory": True,
    },
    "inference": {"sleep_ms_when_empty": 200},
    "rl": {
        "gamma": 0.99,
        "lr": 1e-4,
        "exploration_method": "epsilon",
        "epsilon_start": 1.0,
        "epsilon_end": 0.02,
        "epsilon_decay_steps": 50000,
        "boltzmann_temperature_start": 5.0,
        "boltzmann_temperature_end": 0.5,
        "boltzmann_temperature_decay_steps": 50000,
        "target_update": 1000,
        "grad_clip": 1.0,
        "use_double_dqn": True,
    },
    "record": {"enable": True, "output_dir": "./videos"},
    "log": {"level": "INFO", "file": "./logs/app.log"},
}


def _deep_merge(base: dict, override: dict):
    merged = dict(base)
    for key, value in (override or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path="config/config.yaml"):
    """
    加载 YAML 配置文件
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = _deep_merge(DEFAULT_CONFIG, raw)

    if config["ai"]["continue_frames_num"] <= 0:
        raise ValueError("ai.continue_frames_num must be > 0")
    if config["ai"]["frams_resize"] <= 0:
        raise ValueError("ai.frams_resize must be > 0")
    if config["app"]["train_fps"] <= 0 or config["app"]["recorder_fps"] <= 0:
        raise ValueError("app.train_fps and app.recorder_fps must be > 0")

    logger.info(f"Config loaded from {config_path}")
    return config
