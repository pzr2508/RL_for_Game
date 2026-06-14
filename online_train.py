"""
Online RL training mode: Agent plays the game and learns in real-time.

This module demonstrates how to combine inference (playing) with online training
using a Replay Buffer. The agent:
  1. Captures screen frames -> builds state
  2. Selects action (with exploration)
  3. Executes action in game
  4. Observes reward from learned reward model
  5. Stores transition in replay buffer
  6. Periodically trains from buffer samples
  7. Periodically updates the reward model via TD-consistency

Usage:
    python main.py --mode online
"""

import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rl.agent import RLAgent
from rl.agent import SharedBackbone, AgentConfig
from rl.trainer import OnlineTrainer
from rl.reward_model import RewardModel
from utils.config_loader import load_config
from utils.my_logger import logger


ACTION_NUM = 26


def run_online(runtime, config):
    """
    Online learning loop: inference + training from replay buffer.
    """
    from core.screen_capture import ScreenCaptureThread
    from core.frame_buffer import FrameBuffer
    from core.vision_engine import VisionEngine
    from core.recorder import ScreenRecorder
    from input.keyboard_controller import KeyboardController
    from input.mouse_controller import MouseControllerGame
    from logic.decision import DecisionEngine

    # --- Setup Agent ---
    # Create shared backbone for both online_net and reward model
    agent_cfg_dict = runtime["rl_agent_cfg"]
    shared_backbone = SharedBackbone(
        input_frames=runtime["state_frames"],
        cfg=AgentConfig(
            model_dim=agent_cfg_dict.get("model_dim", 256),
            transformer_layers=agent_cfg_dict.get("transformer_layers", 2),
            transformer_heads=agent_cfg_dict.get("transformer_heads", 8),
            transformer_dropout=agent_cfg_dict.get("transformer_dropout", 0.1),
        ),
    )

    agent = RLAgent(
        num_actions=ACTION_NUM,
        model_input_dime=runtime["state_frames"],
        target_update=runtime["target_update"],
        config=runtime["rl_agent_cfg"],
        backbone=shared_backbone,
    )

    if os.path.exists(runtime["model_path"]):
        logger.info(f"Loading pretrained model: {runtime['model_path']}")
        agent.load(runtime["model_path"], strict=False)

    online_cfg = runtime["online"]

    # --- Setup Learned Reward Model ---
    reward_model_cfg = online_cfg.get("reward_model", {})
    reward_model_path = reward_model_cfg.get("path", "models/reward_model.pth")
    reward_model = RewardModel(
        input_frames=runtime["state_frames"],
        num_actions=ACTION_NUM,
        hidden_dim=reward_model_cfg.get("hidden_dim", 256),
        backbone=shared_backbone,
        lr=reward_model_cfg.get("lr", 3e-4),
        device=None,
    )
    if os.path.exists(reward_model_path):
        reward_model.load(reward_model_path)
        logger.info(f"Loaded pretrained reward model: {reward_model_path}")
        # Restore shared backbone to DQN state.
        # reward_model.load() writes its own backbone weights into shared_backbone,
        # overwriting the DQN weights. target_net.backbone is independent and
        # still holds the DQN checkpoint's backbone state.
        shared_backbone.load_state_dict(agent.target_net.backbone.state_dict())
        logger.info("Restored shared backbone to DQN checkpoint weights")
    else:
        logger.warning(
            f"Reward model not found at {reward_model_path}. "
            "Predictions will be random until pretrained. "
            "Run: python pretrain_reward_model.py"
        )

    reward_model_update_every = reward_model_cfg.get("update_every", 20)
    reward_model_save_every = reward_model_cfg.get("save_every_steps", 200)

    # Cache references for periodic offline_replay refresh
    _offline_cache_for_refresh = None
    _offline_rm_mean = 0.0
    _offline_rm_std = 1.0
    _offline_refresh_interval = max(100, reward_model_save_every * 2)

    # --- Populate offline_replay for anti-forgetting ---
    if len(reward_model.offline_replay) == 0:
        try:
            from data_loader import load_csv_to_cache, compute_reward_stats
            csv_path = runtime["records_csv"]
            if os.path.exists(csv_path):
                cache = load_csv_to_cache(csv_path)
                rm_mean, rm_std = compute_reward_stats(cache)
                _offline_cache_for_refresh = cache
                _offline_rm_mean = rm_mean
                _offline_rm_std = rm_std
                reward_model.populate_offline_replay(
                    cache,
                    img_size=runtime["frame_size"],
                    continue_num=runtime["state_frames"],
                    gap_num=1,
                    max_samples=200,
                    reward_mean=rm_mean,
                    reward_std=rm_std,
                )
        except Exception as e:
            logger.warning(f"Failed to populate offline_replay from CSV: {e}")

    # --- Setup Online Trainer with Replay Buffer ---
    trainer = OnlineTrainer(
        agent=agent,
        buffer_capacity=online_cfg["buffer_capacity"],
        min_buffer_size=online_cfg["min_buffer_size"],
        train_every=online_cfg["train_every"],
        batch_size=online_cfg["batch_size"],
        use_per=online_cfg["use_per"],
        per_alpha=online_cfg.get("per_alpha", 0.6),
        per_beta_start=online_cfg.get("per_beta_start", 0.4),
        per_beta_end=online_cfg.get("per_beta_end", 1.0),
        per_beta_anneal_steps=online_cfg.get("per_beta_anneal_steps", 100000),
    )

    # Optionally load existing buffer
    buffer_path = online_cfg.get("save_buffer_path", "")
    if buffer_path and os.path.exists(buffer_path):
        logger.info(f"Loading replay buffer from: {buffer_path}")
        trainer.load_buffer(buffer_path)

    # --- Setup Game Interaction Components ---
    keyboard = KeyboardController()
    mouse = MouseControllerGame()
    vision = VisionEngine(
        enable=True,
        history_len=runtime["state_frames"],
        out_size=runtime["frame_size"],
    )
    decision = DecisionEngine(keyboard, mouse, agent)

    buffer = FrameBuffer()
    capture_thread = ScreenCaptureThread(buffer, config)
    capture_thread.start()

    recorder = ScreenRecorder(
        runtime["record_enable"],
        runtime["record_output_dir"],
        runtime["recorder_fps"],
        (config["screen"]["width"], config["screen"]["height"]),
    )

    writer = SummaryWriter(runtime["log_dir"])

    frame_interval = max(1, int(runtime["recorder_fps"] / max(runtime["train_fps"], 1)))

    logger.info(
        f"Online learning started. "
        f"buffer_capacity={online_cfg['buffer_capacity']}, "
        f"min_buffer={online_cfg['min_buffer_size']}, "
        f"train_every={online_cfg['train_every']}, "
        f"batch_size={online_cfg['batch_size']}"
    )
    logger.info("Press F5 to enable agent, F6 to disable agent.")

    # --- Main Online Loop ---
    frame_count = 0
    prev_state = None
    prev_action = None
    reward_model_step_count = 0
    last_refresh_step = 0

    try:
        while True:
            frame = buffer.get()
            if frame is None:
                time.sleep(runtime["inference_sleep_ms"] / 1000.0)
                continue

            frame_count += 1
            if frame_count % frame_interval != 0:
                recorder.write(frame)
                continue

            # --- Build current state from vision engine ---
            current_state = vision.process(frame)  # shape: (frames, H, W)

            if current_state is None:
                recorder.write(frame)
                continue

            # --- If we have a previous state+action, compute reward and store transition ---
            if prev_state is not None and prev_action is not None:
                # Use learned reward model to predict reward
                reward = reward_model.predict(prev_state, prev_action)
                # done is always False (no episode termination)
                done = False

                trainer.observe(current_state, reward, done)

                # --- Online update of reward model (TD-consistency) ---
                reward_model_step_count += 1
                if reward_model_step_count % reward_model_update_every == 0:
                    rm_loss = reward_model.update_online(
                        state=prev_state,
                        action=prev_action,
                        q_agent=agent,
                        next_state=current_state,
                        gamma=agent.cfg.gamma,
                        update_every=1,
                        num_train_steps=5,
                    )
                    if rm_loss is not None:
                        writer.add_scalar("online/reward_model_loss", rm_loss, reward_model_step_count)
                        # Periodically refresh offline_replay for diversity
                        if (_offline_cache_for_refresh is not None
                            and reward_model_step_count - last_refresh_step >= _offline_refresh_interval):
                            try:
                                reward_model.refresh_offline_replay(
                                    _offline_cache_for_refresh,
                                    img_size=runtime["frame_size"],
                                    continue_num=runtime["state_frames"],
                                    gap_num=1,
                                    max_samples=200,
                                    keep_ratio=0.6,
                                    reward_mean=_offline_rm_mean,
                                    reward_std=_offline_rm_std,
                                )
                                last_refresh_step = reward_model_step_count
                            except Exception as e:
                                logger.warning(f"Failed to refresh offline_replay: {e}")

                # Save reward model periodically
                if reward_model_step_count % reward_model_save_every == 0:
                    reward_model.save(reward_model_path)

                # --- Train DQN from buffer ---
                result = trainer.maybe_train()
                if result:
                    writer.add_scalar("online/loss", result["loss"], trainer.train_steps)
                    writer.add_scalar("online/q_mean", result["q_mean"], trainer.train_steps)

                # Log predicted reward
                writer.add_scalar("online/predicted_reward", reward, trainer.env_steps)

            # --- Select action (with exploration) ---
            action = trainer.act(current_state, train=True)
            prev_state = current_state
            prev_action = action

            # --- Execute action in game ---
            decision.execute_action(action)

            # --- Log stats ---
            writer.add_scalar("online/buffer_size", trainer.buffer_size, trainer.env_steps)
            writer.add_scalar("online/epsilon", agent._compute_epsilon(), trainer.env_steps)

            recorder.write(frame)

    except KeyboardInterrupt:
        logger.info("Online training interrupted by user.")
    finally:
        # Save final state
        agent.save(runtime["model_path"])
        reward_model.save(reward_model_path)
        if buffer_path:
            trainer.save_buffer(buffer_path)
        capture_thread.stop()
        recorder.release()
        writer.close()
        logger.info(
            f"Online training ended. "
            f"Env steps={trainer.env_steps}, "
            f"Train steps={trainer.train_steps}, "
            f"Buffer size={trainer.buffer_size}, "
            f"Reward model updates={reward_model_step_count}"
        )

