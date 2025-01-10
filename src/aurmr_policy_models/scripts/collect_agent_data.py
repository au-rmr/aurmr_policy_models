import os
import gym
import h5py
import numpy as np
from datetime import datetime

from aurmr_policy_models.utils.config_utils import (
    load_config,
    initialize_agent_from_config,
    initialize_env_from_config
)

def main(cfg):
    # create output dir if it doesn't exist
    os.makedirs(cfg.collection.output, exist_ok=True)

    env = initialize_env_from_config(cfg)
    agent = initialize_agent_from_config(cfg, env)
    filename = f"{cfg.collection.output}/{cfg.collection.collection_name}.hdf5"
    collect_data(env, agent, cfg.collection.num_episodes, cfg.collection.max_steps, filename, cfg.agent.version)
    env.close()

def collect_data(env, agent, num_episodes, max_steps, filename, policy_version):
    """
    Collects data from the environment using the specified agent and saves it in an HDF5 file.
    """
    with h5py.File(filename, 'w') as f:
        f.attrs['policy_version'] = policy_version
        for episode in range(num_episodes):
            states, actions, rewards, dones = [], [], [], []
            obs = env.reset()
            print(f"Starting episode {episode}")
            for step in range(max_steps):
                if env.render_enabled:
                    env.render()

                action = agent.select_action(obs)[0]
                next_state, reward, done, _ = env.step(action)

                states.append(obs['state'].flatten())  # Assuming state can be flattened into a 1D array
                actions.append(action)  # Same assumption for action
                rewards.append(reward)
                dones.append(done)

                state = next_state

                if done:
                    break

            grp = f.create_group(f'episode_{episode}')
            grp.create_dataset('states', data=np.vstack(states).astype(np.float32))  # Stack and convert
            grp.create_dataset('actions', data=np.vstack(actions).astype(np.float32))  # Stack and convert
            grp.create_dataset('rewards', data=np.array(rewards, dtype=np.float32))
            grp.create_dataset('dones', data=np.array(dones, dtype=np.bool_))

if __name__ == '__main__':
    cfg = load_config()
    main(cfg)
