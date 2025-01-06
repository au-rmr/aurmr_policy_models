import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class HDF5StitchedSequenceDataset(Dataset):
    def __init__(self, file_paths, horizon_steps=4, cond_steps=1, max_n_episodes=10000, device="cuda:0"):
        """
        Initialize the dataset with multiple HDF5 file paths.

        Args:
            file_paths (list of str): List of HDF5 file paths.
            horizon_steps (int): Number of steps in each trajectory slice.
            cond_steps (int): Number of conditioning steps.
            max_n_episodes (int): Maximum number of episodes to load from all files combined.
            device (str): Device for tensors (e.g., "cuda:0" or "cpu").
        """
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.device = device
        self.max_n_episodes = max_n_episodes
        self.file_paths = file_paths

        self._load_data()

    def _load_data(self):
        self.traj_lengths = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

        episodes_loaded = 0

        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as f:
                num_episodes_in_file = len(f.keys())

                for i in range(num_episodes_in_file):
                    if episodes_loaded >= self.max_n_episodes:
                        break

                    episode_key = f'episode_{i}'
                    states = f[episode_key]['states'][:]
                    actions = f[episode_key]['actions'][:]
                    rewards = f[episode_key]['rewards'][:]
                    dones = f[episode_key]['dones'][:] if 'dones' in f[episode_key] else np.zeros_like(rewards)

                    self.traj_lengths.append(states.shape[0])
                    self.states.append(torch.from_numpy(states).float())
                    self.actions.append(torch.from_numpy(actions).float())
                    self.rewards.append(torch.from_numpy(rewards).float())
                    self.dones.append(torch.from_numpy(dones).float())

                    episodes_loaded += 1

        self.indices = self._make_indices(self.traj_lengths, self.horizon_steps)
        self.states = torch.cat(self.states).to(self.device)
        self.actions = torch.cat(self.actions).to(self.device)
        self.rewards = torch.cat(self.rewards).to(self.device)
        self.dones = torch.cat(self.dones).to(self.device)

    def _make_indices(self, traj_lengths, horizon_steps):
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            indices.extend([
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ])
            cur_traj_index += traj_length
        return indices

    def __getitem__(self, idx):
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps

        states = self.states[(start - num_before_start):(start + 1)]
        actions = self.actions[start:end]
        rewards = self.rewards[start:end]
        next_states = self.states[start + 1:end + 1]
        dones = self.dones[start:end]

        # Handle the case where next_states length is inconsistent (e.g., at the end of an episode)
        if next_states.shape[0] < actions.shape[0]:
            padding = torch.zeros((actions.shape[0] - next_states.shape[0], *next_states.shape[1:]), device=self.device)
            next_states = torch.cat((next_states, padding), dim=0)

        states = torch.stack([
            states[max(num_before_start - t, 0)] for t in reversed(range(self.cond_steps))
        ])

        conditions = {'state': states}
        next_conditions = {'state': next_states}

        return conditions, actions, rewards, next_conditions, dones

    def __len__(self):
        return len(self.indices)
