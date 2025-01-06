import os
import numpy as np
import torch
import logging
import pickle
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from aurmr_policy_models.trainers.base_trainer import BaseTrainer
from aurmr_policy_models.utils.scheduler_utils import CosineAnnealingWarmupRestarts
from aurmr_policy_models.utils.runtime_utils import Timer
from aurmr_policy_models.utils.reward_utils import RunningRewardScaler

log = logging.getLogger(__name__)

class IDQLTrainer(BaseTrainer):
    def __init__(
        self,
        gamma,
        num_train_iters,
        actor_lr,
        actor_weight_decay,
        actor_lr_scheduler,
        critic_lr,
        critic_weight_decay,
        critic_lr_scheduler,
        buffer_size,
        replay_ratio,
        critic_tau,
        scale_reward_factor,
        eval_sample_num,
        eval_deterministic=False,
        use_expectile_exploration=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.num_train_iters = num_train_iters
        self.eval_sample_num = eval_sample_num
        self.eval_deterministic = eval_deterministic
        self.use_expectile_exploration = use_expectile_exploration
        self.replay_ratio = replay_ratio
        self.critic_tau = critic_tau
        self.scale_reward_factor = scale_reward_factor
        self.buffer_size = buffer_size

        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=actor_lr,
            weight_decay=actor_weight_decay,
        )
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=actor_lr,
            min_lr=actor_lr_scheduler.min_lr,
            warmup_steps=actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_q_optimizer = torch.optim.AdamW(
            self.model.critic_q.parameters(),
            lr=critic_lr,
            weight_decay=critic_weight_decay,
        )
        self.critic_v_optimizer = torch.optim.AdamW(
            self.model.critic_v.parameters(),
            lr=critic_lr,
            weight_decay=critic_weight_decay,
        )
        self.critic_q_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_q_optimizer,
            first_cycle_steps=critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=critic_lr,
            min_lr=critic_lr_scheduler.min_lr,
            warmup_steps=critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_v_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_v_optimizer,
            first_cycle_steps=critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=critic_lr,
            min_lr=critic_lr_scheduler.min_lr,
            warmup_steps=critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        # Initialize replay buffer
        self.obs_buffer = deque(maxlen=buffer_size)
        self.next_obs_buffer = deque(maxlen=buffer_size)
        self.action_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque(maxlen=buffer_size)
        self.terminated_buffer = deque(maxlen=buffer_size)

    def train(self):
        self.model.to(self.device)

        timer = Timer()
        run_results = []
        cnt_train_step = 0

        while self.itr < self.num_train_iters:
            eval_mode = self.itr % self.val_freq == 0
            self.model.eval() if eval_mode else self.model.train()

            prev_obs = self.reset_env_all()

            for step in range(self.num_steps):
                # Select action
                with torch.no_grad():
                    cond = {"state": torch.from_numpy(prev_obs["state"]).float().to(self.device)}
                    actions = (
                        self.model(
                            cond,
                            deterministic=eval_mode and self.eval_deterministic,
                            num_sample=self.eval_sample_num,
                            use_expectile_exploration=self.use_expectile_exploration,
                        )
                        .cpu()
                        .numpy()
                    )

                obs, reward, terminated, truncated, info = self.venv.step(actions)
                done = terminated | truncated

                if not eval_mode:
                    self.obs_buffer.append(prev_obs["state"])
                    self.next_obs_buffer.append(obs["state"])
                    self.action_buffer.append(actions)
                    self.reward_buffer.append(reward * self.scale_reward_factor)
                    self.terminated_buffer.append(terminated)

                prev_obs = obs
                cnt_train_step += self.num_envs if not eval_mode else 0

            if not eval_mode:
                self.update_models()

            self.log_metrics(timer, eval_mode, cnt_train_step, run_results)
            self.itr += 1

    def update_models(self):
        num_batches = int(len(self.obs_buffer) / self.batch_size * self.replay_ratio)

        for _ in range(num_batches):
            indices = np.random.choice(len(self.obs_buffer), self.batch_size, replace=False)

            obs_batch = torch.from_numpy(np.array([self.obs_buffer[i] for i in indices])).float().to(self.device)
            next_obs_batch = torch.from_numpy(np.array([self.next_obs_buffer[i] for i in indices])).float().to(self.device)
            action_batch = torch.from_numpy(np.array([self.action_buffer[i] for i in indices])).float().to(self.device)
            reward_batch = torch.from_numpy(np.array([self.reward_buffer[i] for i in indices])).float().to(self.device)
            terminated_batch = torch.from_numpy(np.array([self.terminated_buffer[i] for i in indices])).float().to(self.device)

            # Update critic Q
            critic_q_loss = self.model.loss_critic_q(
                obs_batch, next_obs_batch, action_batch, reward_batch, terminated_batch, self.gamma
            )
            self.critic_q_optimizer.zero_grad()
            critic_q_loss.backward()
            self.critic_q_optimizer.step()

            # Update critic V
            critic_v_loss = self.model.loss_critic_v(obs_batch, action_batch)
            self.critic_v_optimizer.zero_grad()
            critic_v_loss.backward()
            self.critic_v_optimizer.step()

            # Update actor
            actor_loss = self.model.loss_actor(action_batch, obs_batch)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target network
            self.model.update_target_critic(self.critic_tau)

    def log_metrics(self, timer, eval_mode, cnt_train_step, run_results):
        time_elapsed = timer()
        run_results.append({"itr": self.itr, "step": cnt_train_step, "time": time_elapsed})

        if eval_mode:
            log.info(f"Evaluation at iteration {self.itr}")
        else:
            log.info(f"Training iteration {self.itr}")

        with open(self.result_path, "wb") as f:
            pickle.dump(run_results, f)
