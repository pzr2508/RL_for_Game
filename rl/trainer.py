from rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from utils.my_logger import logger

import numpy as np


class Trainer:
    def __init__(self, agent):
        self.agent = agent
        self.last_state = None
        self.last_action = None

    def step(self, state, reward=None, done=False, train=True, batch_size=32):
        return self.agent.select_action(state, train=train)

    def step_only_train(self, states1, actions, rewards, states2, dones):
        loss, q, _ = self.agent.train_step(states1, actions, rewards, states2, dones)
        return loss, q

    def step_only_inference(self, state):
        action = self.agent.select_action(state, train=False)
        return action

    def reset(self):
        self.last_state = None
        self.last_action = None


class OnlineTrainer:
    """
    Online Trainer with Replay Buffer support.

    Integrates real-time interaction with experience replay for online RL.

    Workflow:
        1. Agent observes state -> selects action
        2. Environment returns reward + next_state + done
        3. Transition stored in replay buffer
        4. Periodically sample from buffer and train

    Usage:
        trainer = OnlineTrainer(agent, buffer_capacity=100000)
        # In the game loop:
        action = trainer.act(state)
        # ... execute action, get reward ...
        trainer.observe(next_state, reward, done)
        loss = trainer.train_step(batch_size=32)
    """

    def __init__(
        self,
        agent,
        buffer_capacity: int = 100000,
        min_buffer_size: int = 1000,
        train_every: int = 4,
        batch_size: int = 32,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_anneal_steps: int = 100000,
        normalize_reward: bool = True,
        state_shape: tuple = (10, 84, 84),
        per_priority_mode: str = "proportional",
    ):
        """
        Args:
            agent: RLAgent instance
            buffer_capacity: Max transitions in replay buffer
            min_buffer_size: Minimum transitions before training starts
            train_every: Train once every N environment steps
            batch_size: Batch size for training from buffer
            use_per: Use Prioritized Experience Replay
            per_alpha: PER priority exponent
            per_beta_start: PER initial importance sampling beta
            per_beta_end: PER final beta
            per_beta_anneal_steps: Steps to anneal beta
            normalize_reward: Whether to normalize rewards in the buffer.
                Set to False when using a learned reward model that already
                outputs normalized rewards (avoids double normalization).
            state_shape: Shape of a single state (frames, H, W).
            per_priority_mode: PER priority mode — "proportional" or "rank".
        """
        self.agent = agent
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.train_every = train_every
        self.use_per = use_per

        if use_per:
            self.buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_end=per_beta_end,
                beta_anneal_steps=per_beta_anneal_steps,
                normalize_reward=normalize_reward,
                priority_mode=per_priority_mode,
            )
        else:
            self.buffer = ReplayBuffer(capacity=buffer_capacity, normalize_reward=normalize_reward,
                                       state_shape=state_shape)

        # Internal state tracking
        self._current_state = None
        self._current_action = None
        self._env_steps = 0
        self._train_steps = 0
        self._episode_reward = 0.0
        self._episode_count = 0

    def act(self, state, train: bool = True) -> int:
        """
        Select an action given the current state.
        Stores state internally for pairing with the subsequent reward.
        """
        self._current_state = state
        action = self.agent.select_action(state, train=train)
        self._current_action = action
        return action

    def observe(self, next_state, reward: float, done: bool):
        """
        Observe the result of taking an action.
        Stores the full transition (s, a, r, s', done) into the replay buffer.
        """
        if self._current_state is None or self._current_action is None:
            return

        self.buffer.push(
            state=self._current_state,
            action=self._current_action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        self._env_steps += 1
        self._episode_reward += reward

        if done:
            self._episode_count += 1
            logger.info(
                f"Episode {self._episode_count} done: "
                f"reward={self._episode_reward:.2f}, "
                f"buffer_size={len(self.buffer)}, "
                f"env_steps={self._env_steps}"
            )
            self._episode_reward = 0.0

        self._current_state = None
        self._current_action = None

    def maybe_train(self) -> dict:
        """
        Train if buffer is ready and it's time to train (every train_every steps).

        Returns:
            dict with 'loss' and 'q_mean' if training occurred, else empty dict.
        """
        if not self.buffer.is_ready(self.min_buffer_size):
            return {}

        if self._env_steps % self.train_every != 0:
            return {}

        return self._do_train_step()

    def force_train(self, n_steps: int = 1) -> dict:
        """Force N training steps regardless of schedule."""
        if not self.buffer.is_ready(self.min_buffer_size):
            logger.warning(
                f"Buffer not ready: {len(self.buffer)}/{self.min_buffer_size} samples"
            )
            return {}

        total_loss = 0.0
        total_q = 0.0
        for _ in range(n_steps):
            result = self._do_train_step()
            if result:
                total_loss += result["loss"]
                total_q += result["q_mean"]

        if n_steps > 0:
            return {"loss": total_loss / n_steps, "q_mean": total_q / n_steps}
        return {}

    def _do_train_step(self) -> dict:
        """Perform one training step from the replay buffer."""
        self._train_steps += 1

        if self.use_per:
            (s1, a, r, s2, done), indices, weights = self.buffer.sample(self.batch_size)
            loss, q_mean, td_errors = self.agent.train_step(s1, a, r, s2, done, weights=weights)
            self.buffer.update_priorities(indices, td_errors)
        else:
            s1, a, r, s2, done = self.buffer.sample(self.batch_size)
            loss, q_mean, _ = self.agent.train_step(s1, a, r, s2, done)

        # Online mode: if Polyak soft sync is disabled, fall back to hard sync
        # at fixed intervals. When use_polyak=True, _polyak_update() in
        # agent.train_step() already handles target network updates every step.
        if not self.agent.cfg.use_polyak and self._train_steps % self.agent.cfg.target_update == 0:
            self.agent.sync_target_network()

        return {"loss": loss, "q_mean": q_mean}

    @property
    def env_steps(self) -> int:
        return self._env_steps

    @property
    def train_steps(self) -> int:
        return self._train_steps

    @property
    def buffer_size(self) -> int:
        return len(self.buffer)

    def save_buffer(self, path: str):
        """Save replay buffer to disk."""
        if hasattr(self.buffer, "save"):
            self.buffer.save(path)

    def load_buffer(self, path: str):
        """Load replay buffer from disk."""
        if hasattr(self.buffer, "load"):
            self.buffer.load(path)