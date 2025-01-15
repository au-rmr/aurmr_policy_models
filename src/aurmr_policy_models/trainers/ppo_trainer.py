import os
import time
import numpy as np
import torch
import logging
import pickle
import math
import einops
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from aurmr_policy_models.trainers.base_trainer import BaseTrainer
from aurmr_policy_models.utils.scheduler_utils import CosineAnnealingWarmupRestarts
from aurmr_policy_models.utils.runtime_utils import Timer
from aurmr_policy_models.utils.reward_utils import RunningRewardScaler
from aurmr_policy_models.utils.gym_utils import render_venv_all


# TODO: upgrade torch and use torch.unravel_index
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


log = logging.getLogger(__name__)

class PPOTrainer(BaseTrainer):
    """
    Proximal Policy Optimization
    """

    def __init__(self, gamma, num_train_iters, num_critic_warmup_iters, num_steps, num_cond_steps,
                 actor_lr, actor_weight_decay, actor_lr_scheduler,
                 critic_lr, critic_weight_decay, critic_lr_scheduler,
                 gae_lambda, target_kl, update_epochs,
                 act_steps, obs_dim, action_dim, batch_size,
                 entropy_coef=0, vf_coef=0,
                 save_full_observations=False,
                 logprob_batch_size=10000,
                 reward_scale_running=False, reward_scale_const=1,
                 learn_eta=False,
                 use_bc_loss=False, bc_loss_coeff=0,
                 max_grad_norm=None,
                #  eta_lr, eta_update_interval, eta_weight_decay,
                 **kwargs):
    # def __init__(self, run_name, env, batch_size, epochs, learning_rate, gamma=0.99, clip_epsilon=0.2, 
    #              entropy_coef=0.01, vf_coef=0.5, max_grad_norm=None, update_epochs=4, 
    #              target_kl=None, output_dir="checkpoints"):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.num_critic_warmup_iters = num_critic_warmup_iters
        self.num_train_iters = num_train_iters
        self.num_cond_steps = num_cond_steps
        self.act_steps = act_steps
        self.reward_horizon = act_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.save_full_observations = save_full_observations
        self.furniture_sparse_reward = False
        self.logprob_batch_size = logprob_batch_size
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.itr = 0

        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),
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

        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=critic_lr,
            weight_decay=critic_weight_decay,
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

        self.reward_scale_running: bool = reward_scale_running
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(self.num_envs)
        self.reward_scale_const = reward_scale_const

        
        self.gae_lambda = gae_lambda
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

        self.use_bc_loss = use_bc_loss
        self.bc_loss_coeff = bc_loss_coeff

        self.max_grad_norm = max_grad_norm
        self.learn_eta = learn_eta
        # Eta - between DDIM (=0 for eval) and DDPM (=1 for training)
        # self.learn_eta = self.model.learn_eta
        # if self.learn_eta:
        #     self.eta_update_interval = eta_update_interval
        #     self.eta_optimizer = torch.optim.AdamW(
        #         self.model.eta.parameters(),
        #         lr=eta_lr,
        #         weight_decay=eta_weight_decay,
        #     )
        #     self.eta_lr_scheduler = CosineAnnealingWarmupRestarts(
        #         self.eta_optimizer,
        #         first_cycle_steps=cfg.train.eta_lr_scheduler.first_cycle_steps,
        #         cycle_mult=1.0,
        #         max_lr=cfg.train.eta_lr,
        #         min_lr=cfg.train.eta_lr_scheduler.min_lr,
        #         warmup_steps=cfg.train.eta_lr_scheduler.warmup_steps,
        #         gamma=1.0,
        #     )

        

    def train(self):

        self.model.cuda()

        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_iter_eval = False
        done_venv = np.zeros((1, self.num_envs))

        while self.itr < self.num_train_iters:
            eval_mode = self.itr % self.val_freq == 0
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            options_venv = [{} for _ in range(self.num_envs)]
            # if self.itr % self.render_freq == 0 and self.render_video:
            #     for env_ind in range(self.n_render):
            #         options_venv[env_ind]["video_path"] = os.path.join(
            #             self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
            #         )

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
            firsts_trajs = np.zeros((self.num_steps + 1, self.num_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv

            # Holder
            obs_trajs = {
                "state": np.zeros(
                    (self.num_steps, self.num_envs, self.num_cond_steps, self.obs_dim)
                )
            }
            chains_trajs = np.zeros(
                (
                    self.num_steps,
                    self.num_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            terminated_trajs = np.zeros((self.num_steps, self.num_envs))
            reward_trajs = np.zeros((self.num_steps, self.num_envs))
            # if self.save_full_observations:  # state-only
            #     obs_full_trajs = np.empty((0, self.num_envs, self.obs_dim))
            #     obs_full_trajs = np.vstack(
            #         (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
            #     )

            # Collect a set of trajectories from env
            for step in range(self.num_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.num_steps}")

                # Select action
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"])
                        .float()
                        .to(self.device)
                    }
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=True,
                    )
                    output_venv = (
                        samples.trajectories.cpu().numpy()
                    )  # n_env x horizon x act
                    chains_venv = (
                        samples.chains.cpu().numpy()
                    )  # n_env x denoising x horizon x act
                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                (
                    obs_venv,
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                    info_venv,
                ) = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                if self.save_full_observations:  # state-only
                    obs_full_venv = np.array(
                        [info["full_obs"]["state"] for info in info_venv]
                    )  # num_envs x act_steps x obs_dim
                    obs_full_trajs = np.vstack(
                        (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                    )
                obs_trajs["state"][step] = prev_obs_venv["state"]
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

                if self.save_videos:
                    self.video_writer.render_and_write_frames()

                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.num_envs * self.act_steps if not eval_mode else 0
            
            if self.save_videos and self.itr < self.num_train_iters -1:
                self.video_writer.next_episode()
            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.num_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                if (
                    self.furniture_sparse_reward
                ):  # only for furniture tasks, where reward only occurs in one env step
                    episode_best_reward = episode_reward
                else:
                    episode_best_reward = np.array(
                        [
                            np.max(reward_traj) / self.act_steps
                            for reward_traj in reward_trajs_split
                        ]
                    )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Update models
            if not eval_mode:
                with torch.no_grad():
                    obs_trajs["state"] = (
                        torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                    )

                    # Calculate value and logprobs - split into batches to prevent out of memory
                    num_split = math.ceil(
                        self.num_envs * self.num_steps / self.logprob_batch_size
                    )
                    obs_ts = [{} for _ in range(num_split)]
                    obs_k = einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                    obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                    for i, obs_t in enumerate(obs_ts_k):
                        obs_ts[i]["state"] = obs_t
                    values_trajs = np.empty((0, self.num_envs))
                    for obs in obs_ts:
                        values = self.model.critic(obs).cpu().numpy().flatten()
                        values_trajs = np.vstack(
                            (values_trajs, values.reshape(-1, self.num_envs))
                        )
                    chains_t = einops.rearrange(
                        torch.from_numpy(chains_trajs).float().to(self.device),
                        "s e t h d -> (s e) t h d",
                    )
                    chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
                    logprobs_trajs = np.empty(
                        (
                            0,
                            self.model.ft_denoising_steps,
                            self.horizon_steps,
                            self.action_dim,
                        )
                    )
                    for obs, chains in zip(obs_ts, chains_ts):
                        logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
                        logprobs_trajs = np.vstack(
                            (
                                logprobs_trajs,
                                logprobs.reshape(-1, *logprobs_trajs.shape[1:]),
                            )
                        )

                    # normalize reward with running variance if specified
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T

                    # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
                    obs_venv_ts = {
                        "state": torch.from_numpy(obs_venv["state"])
                        .float()
                        .to(self.device)
                    }
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextvalues = (
                                self.model.critic(obs_venv_ts)
                                .reshape(1, -1)
                                .cpu()
                                .numpy()
                            )
                        else:
                            nextvalues = values_trajs[t + 1]
                        nonterminal = 1.0 - terminated_trajs[t]
                        # delta = r + gamma*V(st+1) - V(st)
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * nextvalues * nonterminal
                            - values_trajs[t]
                        )
                        # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                        advantages_trajs[t] = lastgaelam = (
                            delta
                            + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                        )
                    returns_trajs = advantages_trajs + values_trajs

                # k for environment step
                obs_k = {
                    "state": einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                }
                chains_k = einops.rearrange(
                    torch.tensor(chains_trajs, device=self.device).float(),
                    "s e t h d -> (s e) t h d",
                )
                returns_k = (
                    torch.tensor(returns_trajs, device=self.device).float().reshape(-1)
                )
                values_k = (
                    torch.tensor(values_trajs, device=self.device).float().reshape(-1)
                )
                advantages_k = (
                    torch.tensor(advantages_trajs, device=self.device)
                    .float()
                    .reshape(-1)
                )
                logprobs_k = torch.tensor(logprobs_trajs, device=self.device).float()

                # Update policy and critic
                total_steps = self.num_steps * self.num_envs * self.model.ft_denoising_steps
                clipfracs = []
                for update_epoch in range(self.update_epochs):
                    # for each epoch, go through all data in batches
                    flag_break = False
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        batch_inds_b, denoising_inds_b = unravel_index(
                            inds_b,
                            (self.num_steps * self.num_envs, self.model.ft_denoising_steps),
                        )
                        obs_b = {"state": obs_k["state"][batch_inds_b]}
                        chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]
                        chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]
                        returns_b = returns_k[batch_inds_b]
                        values_b = values_k[batch_inds_b]
                        advantages_b = advantages_k[batch_inds_b]
                        logprobs_b = logprobs_k[batch_inds_b, denoising_inds_b]

                        # get loss
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            eta,
                        ) = self.model.loss(
                            obs_b,
                            chains_prev_b,
                            chains_next_b,
                            denoising_inds_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.entropy_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs += [clipfrac]

                        # update policy and critic
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        if self.learn_eta:
                            self.eta_optimizer.zero_grad()
                        loss.backward()
                        if self.itr >= self.num_critic_warmup_iters:
                            if self.max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.actor_ft.parameters(), self.max_grad_norm
                                )
                            self.actor_optimizer.step()
                            if self.learn_eta and batch % self.eta_update_interval == 0:
                                self.eta_optimizer.step()
                        self.critic_optimizer.step()
                        log.info(
                            f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                        )

                        # Stop gradient update if KL difference reaches target
                        if self.target_kl is not None and approx_kl > self.target_kl:
                            flag_break = True
                            break
                    if flag_break:
                        break

                # Explained variation of future rewards using value function
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

            # Plot state trajectories (only in D3IL)
            # if (
            #     self.itr % self.render_freq == 0
            #     and self.n_render > 0
            #     and self.traj_plotter is not None
            # ):
            #     self.traj_plotter(
            #         obs_full_trajs=obs_full_trajs,
            #         n_render=self.n_render,
            #         max_episode_steps=self.max_episode_steps,
            #         render_dir=self.render_dir,
            #         itr=self.itr,
            #     )

            # Update lr, min_sampling_std
            if self.itr >= self.num_critic_warmup_iters:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.num_train_iters - 1:
                self.save_checkpoint(self.model, self.itr)

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["chains_trajs"] = chains_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    print(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    else:
                        print({
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            })
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | eta {eta:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "loss": loss,
                                "pg loss": pg_loss,
                                "value loss": v_loss,
                                "bc loss": bc_loss,
                                "eta": eta,
                                "approx kl": approx_kl,
                                "ratio": ratio,
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                                "diffusion - min sampling std": diffusion_min_sampling_std,
                                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                                "critic lr": self.critic_optimizer.param_groups[0][
                                    "lr"
                                ],
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1


        

        # Optimizers
        # optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)

        # obs = self.env.reset()

        # for epoch in range(self.epochs):
        #     model.train()
        #     total_loss, total_pg_loss, total_v_loss, total_entropy = 0, 0, 0, 0
        #     epoch_start_time = time.time()

        #     observations, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        #     # Collect trajectories
        #     for _ in range(self.batch_size):
        #         obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).cuda()
        #         with torch.no_grad():
        #             action, log_prob, value = model.act(obs_tensor)

        #         next_obs, reward, done, _ = self.env.step(action.cpu().numpy())

        #         observations.append(obs)
        #         actions.append(action.cpu().numpy())
        #         log_probs.append(log_prob.cpu().numpy())
        #         rewards.append(reward)
        #         values.append(value.cpu().numpy())
        #         dones.append(done)

        #         obs = next_obs if not done else self.env.reset()

        #     # Process trajectories
        #     returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)

        #     # Convert to tensors
        #     observations = torch.tensor(observations, dtype=torch.float32).cuda()
        #     actions = torch.tensor(actions, dtype=torch.float32).cuda()
        #     old_log_probs = torch.tensor(log_probs, dtype=torch.float32).cuda()
        #     returns = torch.tensor(returns, dtype=torch.float32).cuda()
        #     advantages = torch.tensor(advantages, dtype=torch.float32).cuda()

        #     # Update policy and value function
        #     for _ in range(self.update_epochs):
        #         new_log_probs, entropy, new_values = model.evaluate(observations, actions)

        #         # Compute losses
        #         ratio = torch.exp(new_log_probs - old_log_probs)
        #         surr1 = ratio * advantages
        #         surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        #         pg_loss = -torch.min(surr1, surr2).mean()
        #         v_loss = torch.nn.functional.mse_loss(new_values.squeeze(-1), returns)
        #         entropy_loss = -entropy.mean()
        #         loss = pg_loss + self.vf_coef * v_loss + self.entropy_coef * entropy_loss

        #         # Optimize
        #         optimizer.zero_grad()
        #         loss.backward()
        #         if self.max_grad_norm is not None:
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        #         optimizer.step()

        #         # Track losses
        #         total_loss += loss.item()
        #         total_pg_loss += pg_loss.item()
        #         total_v_loss += v_loss.item()
        #         total_entropy += entropy_loss.item()

        #     # Log average losses
        #     avg_loss = total_loss / self.update_epochs
        #     avg_pg_loss = total_pg_loss / self.update_epochs
        #     avg_v_loss = total_v_loss / self.update_epochs
        #     avg_entropy = total_entropy / self.update_epochs
        #     epoch_duration = time.time() - epoch_start_time

        #     self.writer.add_scalar("Loss/Total", avg_loss, epoch)
        #     self.writer.add_scalar("Loss/Policy", avg_pg_loss, epoch)
        #     self.writer.add_scalar("Loss/Value", avg_v_loss, epoch)
        #     self.writer.add_scalar("Loss/Entropy", avg_entropy, epoch)
        #     self.writer.add_scalar("Time/Epoch", epoch_duration, epoch)

        #     print(f"Epoch {epoch}: Avg Loss = {avg_loss}, Avg Policy Loss = {avg_pg_loss}, Avg Value Loss = {avg_v_loss}, Avg Entropy = {avg_entropy}, Duration = {epoch_duration:.2f} seconds")

        #     # Save checkpoints
        #     if (epoch + 1) % 5 == 0:
        #         self.save_checkpoint(model, epoch)

        # # Save final model
        # self.save_final_model(model)
        # self.writer.close()

    def compute_returns_and_advantages(self, rewards, values, dones):
        returns = []
        advantages = []
        gae = 0
        next_value = 0

        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * next_value * mask - values[step]
            gae = delta + self.gamma * 0.95 * mask * gae
            advantages.insert(0, gae)
            next_value = values[step]
            returns.insert(0, gae + values[step])

        return returns, advantages
