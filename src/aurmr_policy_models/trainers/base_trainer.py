from abc import ABC, abstractmethod
import os
import torch
import pickle
import einops
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import numpy as np

log = logging.getLogger(__name__)

from aurmr_policy_models.utils.gym_utils.video_writer import AsyncVectorEnvVideoWriter
from aurmr_policy_models.utils.gym_utils import (
    make_async_vector_env,
    reset_venv_all,
    reset_venv,
    set_venv_seeds,
)


class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.
    """
    def __init__(self, model, run_name, output_dir="checkpoints", seed=42, device='cuda:0',
                 env=None, num_envs=0, reset_at_iteration=True,
                 horizon_steps=1, cond_steps=1, train_dataset=None, val_dataset=None,
                 log_freq=1, val_freq=1, save_model_freq=1, best_reward_threshold_for_success=-100,
                 save_trajs=False, save_videos=False, video_fps=30,
                 use_wandb=False):
        # self.config = config
        self.model = model
        self.output_dir = output_dir
        self.result_path = os.path.join(self.output_dir, "results.npy")
        self.seed = seed
        self.device = device

        self.run_name = run_name
        # self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs', run_name))

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.num_envs = num_envs
        self.env = env
        self.venv = None
        self.reset_at_iteration = reset_at_iteration

        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.val_freq = val_freq
        self.save_model_freq = save_model_freq
        self.log_freq = log_freq

        self.best_reward_threshold_for_success = best_reward_threshold_for_success
        # self.best_reward_threshold_for_success = (
        #     len(self.venv.pairs_to_assemble)
        #     if env_type == "furniture"
        #     else cfg.env.best_reward_threshold_for_success
        # )


        self.save_trajs = save_trajs
        self.use_wandb = use_wandb
        self.save_videos = save_videos
        self.video_fps = video_fps

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def __configure__(self, cfg):
        self.cfg = cfg
        if self.num_envs > 0:
            self.venv = make_async_vector_env(cfg.env, self.num_envs, True, self.cond_steps, self.horizon_steps)
            set_venv_seeds(self.venv, cfg.seed)
            reset_venv_all(self.venv)
        
        self.video_writer = None
        if self.save_videos:
            video_path = os.path.join(self.output_dir, "videos")
            os.makedirs(video_path, exist_ok=True)
            self.video_writer = AsyncVectorEnvVideoWriter(self.venv, video_path, self.video_fps)

    @abstractmethod
    def train(self):
        """
        Abstract method to train the model.
        To be implemented by subclasses with specific learning algorithms.
        """
        pass

    def save_checkpoint(self, model, epoch):
        """
        Save a checkpoint of the model.

        Args:
            model (torch.nn.Module): The model to save.
            epoch (int): The current epoch number.
        """
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({'model': model.state_dict()}, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def save_final_model(self, model):
        """
        Save the final model after training is complete.

        Args:
            model (torch.nn.Module): The model to save.
        """
        final_model_path = os.path.join(self.output_dir, "final_model.pt")
        torch.save({'model': model.state_dict()}, final_model_path)
        print(f"Final model saved: {final_model_path}")
    
    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        obs_venv = reset_venv_all(self.venv, verbose=verbose, options_venv=options_venv, **kwargs)
        return obs_venv

    def reset_env(self, env_ind, verbose=False):
        obs = reset_venv(self.venv, env_ind, verbose)
        return obs


