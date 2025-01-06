import os
import pickle
import numpy as np
import torch
import logging
import einops
from collections import deque
from torch.utils.data import DataLoader
from tqdm import tqdm
from aurmr_policy_models.trainers.base_trainer import BaseTrainer
from aurmr_policy_models.utils.scheduler_utils import CosineAnnealingWarmupRestarts
from aurmr_policy_models.utils.runtime_utils import Timer

log = logging.getLogger(__name__)

class TrainRLPDAgent(BaseTrainer):
    def __init__(self, gamma, num_train_iters, num_steps, act_steps,
                 actor_lr, actor_weight_decay, actor_lr_scheduler,
                 critic_lr, critic_weight_decay, critic_lr_scheduler,
                 target_ema_rate, scale_reward_factor,
                 critic_num_update, buffer_size, batch_size,
                 init_temperature, target_entropy,
                 n_explore_steps, n_eval_episode, **kwargs):
        super().__init__(**kwargs)

        self.gamma = gamma
        self.num_train_iters = num_train_iters
        self.num_steps = num_steps
        self.act_steps = act_steps
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_ema_rate = target_ema_rate
        self.scale_reward_factor = scale_reward_factor
        self.critic_num_update = critic_num_update
        self.n_explore_steps = n_explore_steps
        self.n_eval_episode = n_eval_episode

        self.log_alpha = torch.tensor(np.log(init_temperature), device=self.device, requires_grad=True)
        self.target_entropy = target_entropy

        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(), lr=actor_lr, weight_decay=actor_weight_decay
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

        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(), lr=critic_lr, weight_decay=critic_weight_decay
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=critic_lr,
            min_lr=critic_lr_scheduler.min_lr,
            warmup_steps=critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=actor_lr)

        self.obs_buffer = deque(maxlen=buffer_size)
        self.next_obs_buffer = deque(maxlen=buffer_size)
        self.action_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque(maxlen=buffer_size)
        self.terminated_buffer = deque(maxlen=buffer_size)

    def train(self):
        timer = Timer()
        run_results = []
        cnt_train_step = 0

        while self.itr < self.num_train_iters:
            eval_mode = self.itr % self.val_freq == 0 and self.itr >= self.n_explore_steps
            self.model.eval() if eval_mode else self.model.train()

            # Reset environments if required
            if self.itr == 0 or eval_mode or self.reset_at_iteration:
                prev_obs_venv = self.reset_env_all()

            for step in range(self.num_steps):
                # Select action
                if self.itr < self.n_explore_steps:
                    action_venv = self.venv.action_space.sample()
                else:
                    with torch.no_grad():
                        cond = {"state": torch.tensor(prev_obs_venv["state"]).float().to(self.device)}
                        action_venv = (
                            self.model(cond, deterministic=eval_mode).cpu().numpy()[:, :self.act_steps]
                        )

                # Environment step
                obs_venv, reward_venv, terminated_venv, truncated_venv, _ = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv

                # Add to buffers (training mode only)
                if not eval_mode:
                    for i in range(self.num_envs):
                        self.obs_buffer.append(prev_obs_venv["state"][i])
                        self.next_obs_buffer.append(obs_venv["state"][i] if not truncated_venv[i] else None)
                        self.action_buffer.append(action_venv[i])
                        self.reward_buffer.append(reward_venv[i] * self.scale_reward_factor)
                        self.terminated_buffer.append(done_venv[i])

                prev_obs_venv = obs_venv

            # Model updates
            if not eval_mode and self.itr >= self.n_explore_steps:
                for _ in range(self.critic_num_update):
                    self._update_critic()

                self._update_actor()

                self._update_temperature()

            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

            # Save model and log results
            if self.itr % self.save_model_freq == 0:
                self.save_checkpoint(self.model, self.itr)

            self.itr += 1

    def _update_critic(self):
        # Sample from buffer
        obs_b, next_obs_b, actions_b, rewards_b, terminated_b = self._sample_batch()

        # Update critic
        alpha = self.log_alpha.exp().item()
        loss_critic = self.model.loss_critic(obs_b, next_obs_b, actions_b, rewards_b, terminated_b, self.gamma, alpha)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        self.model.update_target_critic(self.target_ema_rate)

    def _update_actor(self):
        obs_b, _, _, _, _ = self._sample_batch()
        alpha = self.log_alpha.exp().item()
        loss_actor = self.model.loss_actor(obs_b, alpha)

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

    def _update_temperature(self):
        obs_b, _, _, _, _ = self._sample_batch()
        alpha_loss = self.model.loss_temperature(obs_b, self.log_alpha.exp(), self.target_entropy)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def _sample_batch(self):
        inds = np.random.choice(len(self.obs_buffer), self.batch_size)
        obs_b = torch.tensor([self.obs_buffer[i] for i in inds]).float().to(self.device)
        next_obs_b = torch.tensor([self.next_obs_buffer[i] for i in inds]).float().to(self.device)
        actions_b = torch.tensor([self.action_buffer[i] for i in inds]).float().to(self.device)
        rewards_b = torch.tensor([self.reward_buffer[i] for i in inds]).float().to(self.device)
        terminated_b = torch.tensor([self.terminated_buffer[i] for i in inds]).float().to(self.device)
        return obs_b, next_obs_b, actions_b, rewards_b, terminated_b
