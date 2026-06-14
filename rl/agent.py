import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.my_logger import logger


@dataclass
class AgentConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    epsilon_start: float = 0.15
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 200000
    target_update: int = 1000
    grad_clip: float = 1.0
    use_double_dqn: bool = True
    model_dim: int = 256
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_dropout: float = 0.1
    inference_topk: int = 3
    inference_log_every: int = 30
    inference_tie_delta: float = 0.0
    inference_tie_topk: int = 3
    inference_tie_temperature: float = 5.0
    gpu_ids: Tuple[int, ...] = ()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, model_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]


class SharedBackbone(nn.Module):
    """Full temporal feature backbone — the entire forward_features logic.

    Encapsulates: FrameEncoder -> PositionalEncoding -> TransformerEncoder ->
    temporal attention pooling + last-frame -> Fusion (512-dim).

    A single instance can be shared between DuelingDQN (online_net) and
    RewardNet. Both optimizers update the same underlying parameters.
    """
    def __init__(self, input_frames: int, cfg: AgentConfig):
        super().__init__()
        self.model_dim = cfg.model_dim
        self.frame_encoder = FrameEncoder(cfg.model_dim)
        self.pos_encoder = TemporalPositionalEncoding(cfg.model_dim, input_frames)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.model_dim * 4,
            dropout=cfg.transformer_dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)

        self.temporal_attn = nn.Sequential(
            nn.Linear(cfg.model_dim, cfg.model_dim // 2),
            nn.Tanh(),
            nn.Linear(cfg.model_dim // 2, 1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cfg.model_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.transformer_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the 512-dim fused feature vector.

        Args:
            x: (batch, seq_len, H, W) normalized grayscale frames

        Returns:
            (batch, 512) fused feature
        """
        batch_size, seq_len, height, width = x.shape
        x = x.view(batch_size * seq_len, 1, height, width)
        frame_feat = self.frame_encoder(x)
        frame_feat = frame_feat.view(batch_size, seq_len, self.model_dim)

        temporal_feat = self.pos_encoder(frame_feat)
        temporal_feat = self.temporal_encoder(temporal_feat)

        attn_logits = self.temporal_attn(temporal_feat)
        attn_weight = torch.softmax(attn_logits, dim=1)
        context_feat = (attn_weight * temporal_feat).sum(dim=1)
        last_feat = temporal_feat[:, -1, :]

        fused_feat = self.fusion(torch.cat([context_feat, last_feat], dim=1))
        return fused_feat


class FrameEncoder(nn.Module):
    """Shared CNN backbone for per-frame feature extraction.

    Used by both DuelingDQN and RewardNet to ensure consistent
    visual feature quality (with ResidualBlocks).
    Each caller instantiates its own copy — no parameter sharing.
    """
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frame-level features.

        Args:
            x: (B, 1, H, W) or (B, H, W) grayscale frame batch

        Returns:
            (B, out_dim) feature vectors
        """
        feat = self.cnn(x)
        feat = self.proj(feat)
        return feat


class DuelingDQN(nn.Module):
    def __init__(self, input_frames: int, num_actions: int, cfg: AgentConfig,
                 backbone: Optional[SharedBackbone] = None):
        super().__init__()
        self.input_frames = input_frames
        self.model_dim = cfg.model_dim

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = SharedBackbone(input_frames, cfg)

        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused_feat = self.forward_features(x)
        value = self.value_stream(fused_feat)
        advantage = self.adv_stream(fused_feat)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the shared backbone (CNN + Transformer + fusion)."""
        return self.backbone(x)


class RLAgent:
    def __init__(
        self,
        num_actions: int,
        model_input_dime: int,
        target_update: int = 1000,
        device: Optional[str] = None,
        config: Optional[dict] = None,
        backbone: Optional[SharedBackbone] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions

        def _normalize_gpu_ids(ids) -> Tuple[int, ...]:
            if ids is None:
                return ()
            if isinstance(ids, str):
                parts = [x.strip() for x in ids.split(",") if x.strip()]
                return tuple(int(x) for x in parts)
            if isinstance(ids, Sequence):
                return tuple(int(x) for x in ids)
            return ()

        cfg = config or {}
        self.cfg = AgentConfig(
            gamma=cfg.get("gamma", 0.99),
            lr=cfg.get("lr", 1e-4),
            epsilon_start=cfg.get("epsilon_start", 0.15),
            epsilon_end=cfg.get("epsilon_end", 0.02),
            epsilon_decay_steps=cfg.get("epsilon_decay_steps", 200000),
            target_update=cfg.get("target_update", target_update),
            grad_clip=cfg.get("grad_clip", 1.0),
            use_double_dqn=cfg.get("use_double_dqn", True),
            model_dim=cfg.get("model_dim", 256),
            transformer_layers=cfg.get("transformer_layers", 2),
            transformer_heads=cfg.get("transformer_heads", 8),
            transformer_dropout=cfg.get("transformer_dropout", 0.1),
            inference_topk=cfg.get("inference_topk", 3),
            inference_log_every=cfg.get("inference_log_every", 30),
            inference_tie_delta=cfg.get("inference_tie_delta", 0.0),
            inference_tie_topk=cfg.get("inference_tie_topk", 3),
            inference_tie_temperature=cfg.get("inference_tie_temperature", 5.0),
            gpu_ids=_normalize_gpu_ids(cfg.get("gpu_ids", ())),
        )

        if self.cfg.model_dim % self.cfg.transformer_heads != 0:
            raise ValueError("model_dim must be divisible by transformer_heads")

        if torch.cuda.is_available() and self.cfg.gpu_ids:
            gpu_count = torch.cuda.device_count()
            valid_gpu_ids = tuple(gid for gid in self.cfg.gpu_ids if 0 <= gid < gpu_count)
            if not valid_gpu_ids:
                logger.warning(
                    f"Configured gpu_ids={self.cfg.gpu_ids} are invalid for current machine, fallback to default device {self.device}."
                )
                self.multi_gpu_ids = ()
            else:
                self.multi_gpu_ids = valid_gpu_ids
                self.device = f"cuda:{self.multi_gpu_ids[0]}"
        else:
            self.multi_gpu_ids = ()

        self.online_net = DuelingDQN(model_input_dime, num_actions, self.cfg,
                                     backbone=backbone).to(self.device)
        self.target_net = DuelingDQN(model_input_dime, num_actions, self.cfg).to(self.device)

        if len(self.multi_gpu_ids) > 1:
            self.online_net = nn.DataParallel(self.online_net, device_ids=list(self.multi_gpu_ids), output_device=self.multi_gpu_ids[0])
            self.target_net = nn.DataParallel(self.target_net, device_ids=list(self.multi_gpu_ids), output_device=self.multi_gpu_ids[0])
            logger.info(f"Enable DataParallel training on GPUs: {self.multi_gpu_ids}")

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.cfg.lr)
        self.update_counter = 0
        self.total_env_steps = 0
        self.inference_steps = 0

    def _compute_epsilon(self) -> float:
        progress = min(1.0, self.total_env_steps / float(self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + (self.cfg.epsilon_end - self.cfg.epsilon_start) * progress

    def set_train_mode(self):
        """Set online_net to training mode. Call once before the training loop."""
        self.online_net.train()

    def set_eval_mode(self):
        """Set online_net to evaluation mode. Call once before the inference loop."""
        self.online_net.eval()

    def select_action(self, state: np.ndarray, train: bool = True) -> int:
        if state is None:
            return 0

        self.total_env_steps += 1 if train else 0
        epsilon = self._compute_epsilon() if train else 0.0

        if train and random.random() < epsilon:
            return random.randrange(self.num_actions)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)

        q_row = q_values.squeeze(0)
        action = int(q_row.argmax(dim=0).item())
        tie_break_applied = False
        tie_margin = None

        if not train and self.cfg.inference_tie_delta > 0 and self.num_actions >= 2:
            top2_vals, _ = torch.topk(q_row, k=2, dim=0)
            tie_margin = float((top2_vals[0] - top2_vals[1]).item())
            if tie_margin < float(self.cfg.inference_tie_delta):
                tie_topk = max(2, min(int(self.cfg.inference_tie_topk), self.num_actions))
                cand_vals, cand_idx = torch.topk(q_row, k=tie_topk, dim=0)
                temperature = float(self.cfg.inference_tie_temperature)
                if temperature > 0:
                    scaled_logits = (cand_vals - cand_vals.max()) / temperature
                    cand_probs = torch.softmax(scaled_logits, dim=0)
                else:
                    cand_probs = torch.ones_like(cand_vals) / float(tie_topk)
                sampled_pos = int(torch.multinomial(cand_probs, num_samples=1).item())
                action = int(cand_idx[sampled_pos].item())
                tie_break_applied = True

        if not train:
            self.inference_steps += 1
            should_log = self.cfg.inference_log_every > 0 and (self.inference_steps % self.cfg.inference_log_every == 0)
            if should_log:
                topk = max(1, min(self.cfg.inference_topk, self.num_actions))
                top_vals, top_idx = torch.topk(q_values.squeeze(0), k=topk, dim=0)
                top_items = ", ".join(
                    f"{int(idx)}:{float(val):.4f}"
                    for idx, val in zip(top_idx.tolist(), top_vals.tolist())
                )
                tie_info = ""
                if self.cfg.inference_tie_delta > 0 and tie_margin is not None:
                    tie_info = f"; margin12={tie_margin:.4f}; tie_break={tie_break_applied}"
                logger.info(f"inference_q_top{topk} -> {top_items}; selected={action}{tie_info}")

        return action

    def train_step(self, s1: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                   s2: torch.Tensor, done: torch.Tensor,
                   weights: Optional[torch.Tensor] = None):

        s1 = s1.to(self.device, dtype=torch.float32)
        s2 = s2.to(self.device, dtype=torch.float32)
        a = a.to(self.device, dtype=torch.long).view(-1)
        r = r.to(self.device, dtype=torch.float32).view(-1)
        done = done.to(self.device, dtype=torch.float32).view(-1)

        q_all = self.online_net(s1)
        # logger.info(f"s shape: {s1.shape}, q_all shape: {q_all.shape}, a shape: {a.shape}")
        # logger.info(f"q_all: {q_all}, a: {a}")
        q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.use_double_dqn:
                next_actions = self.online_net(s2).argmax(dim=1, keepdim=True)
                next_q = self.target_net(s2).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(s2).max(dim=1)[0]

            target_q = r + self.cfg.gamma * next_q * (1.0 - done)

        if weights is not None:
            weights = weights.to(self.device, dtype=torch.float32).view(-1)
            per_sample_loss = F.smooth_l1_loss(q_sa, target_q, reduction='none')
            loss = (per_sample_loss * weights).mean()
        else:
            loss = F.smooth_l1_loss(q_sa, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        self.update_counter += 1

        with torch.no_grad():
            per_sample_td = (target_q - q_sa).abs().detach().cpu().numpy()
        return float(loss.item()), float(q_sa.mean().item()), per_sample_td

    def sync_target_network(self):
        """Copy online_net weights to target_net. Should be called every target_update epochs."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        logger.info(f"Target network synced at update_counter={self.update_counter}")

    def save(self, path: str):
        online_to_save = self.online_net.module if isinstance(self.online_net, nn.DataParallel) else self.online_net
        target_to_save = self.target_net.module if isinstance(self.target_net, nn.DataParallel) else self.target_net
        torch.save(
            {
                "online_net": online_to_save.state_dict(),
                "target_net": target_to_save.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_counter": self.update_counter,
                "total_env_steps": self.total_env_steps,
                "agent_config": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str, strict: bool = True):
        checkpoint = torch.load(path, map_location=self.device)

        def _load_state_with_fallback(model: nn.Module, state_dict: dict, model_name: str):
            target_model = model.module if isinstance(model, nn.DataParallel) else model

            if state_dict:
                ckpt_has_module = next(iter(state_dict)).startswith("module.")
                model_has_module = next(iter(target_model.state_dict())).startswith("module.")
                if ckpt_has_module and not model_has_module:
                    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                elif model_has_module and not ckpt_has_module:
                    state_dict = {f"module.{k}": v for k, v in state_dict.items()}

            try:
                target_model.load_state_dict(state_dict)
                return
            except RuntimeError as err:
                if strict:
                    raise RuntimeError(
                        f"{model_name} architecture mismatch while loading '{path}'. "
                        f"Please retrain the model with current network definition."
                    ) from err

                current = target_model.state_dict()
                matched = {
                    k: v
                    for k, v in state_dict.items()
                    if k in current and current[k].shape == v.shape
                }
                if not matched:
                    logger.warning(
                        f"No compatible weights found for {model_name} from checkpoint: {path}. "
                        "Training will continue from scratch."
                    )
                    return

                current.update(matched)
                target_model.load_state_dict(current)
                logger.warning(
                    f"Partially loaded {len(matched)}/{len(current)} tensors into {model_name} from checkpoint: {path}."
                )

        if "online_net" in checkpoint:
            _load_state_with_fallback(self.online_net, checkpoint["online_net"], "online_net")
            _load_state_with_fallback(
                self.target_net,
                checkpoint.get("target_net", checkpoint["online_net"]),
                "target_net",
            )
        elif "q_net" in checkpoint:
            logger.warning("Loaded legacy DQN key format; this checkpoint may not match DuelingDQN architecture.")
            _load_state_with_fallback(self.online_net, checkpoint["q_net"], "online_net")
            _load_state_with_fallback(
                self.target_net,
                checkpoint.get("target_net", checkpoint["q_net"]),
                "target_net",
            )
        else:
            raise KeyError("Invalid checkpoint format: expected 'online_net' or legacy 'q_net'.")

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.update_counter = int(checkpoint.get("update_counter", 0))
        self.total_env_steps = int(checkpoint.get("total_env_steps", 0))
        self.online_net.eval()
        self.target_net.eval()
