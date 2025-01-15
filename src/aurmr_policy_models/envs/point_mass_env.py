import numpy as np
import cv2
import gym
from gym import spaces
import matplotlib.pyplot as plt

class PointMassEnv(gym.Env):
    """
    A 2D environment with a point mass that moves based on continuous acceleration inputs.
    The state includes the position, velocity, and goal position. Obstacles penalize the agent upon collision.
    The environment can return observations as either ground truth or image representation.
    """

    def __init__(self, env_name='point_mass_env', screen_size=5, render_size=255, obstacles=None, render=False, state_mode='ground_truth'):
        super(PointMassEnv, self).__init__()

        self.env_name = env_name
        self.screen_size = screen_size
        self.obstacles = obstacles if obstacles else []
        self.render_enabled = render
        self.state_mode = state_mode  # 'ground_truth' or 'image'

        # Define action space: Continuous acceleration in x and y
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Define observation space
        # self.obs_keys = low_dim_keys
        self.observation_space = spaces.Dict()
        # obs_example_full = self.env.get_observation()
        # obs_example = np.concatenate(
        #     [obs_example_full[key] for key in self.obs_keys], axis=0
        # )
        # low = np.full_like(obs_example, fill_value=-1)
        # high = np.full_like(obs_example, fill_value=1)
        self.observation_space["state"] = spaces.Box(
            # low=low,
            # high=high,
            # shape=low.shape,
            # dtype=np.float32,
            low=np.array([0, 0, -np.inf, -np.inf, 0, 0]),
            high=np.array([screen_size, screen_size, np.inf, np.inf, screen_size, screen_size]),
            dtype=np.float32
        )
        # if state_mode == 'ground_truth':
        #     self.observation_space = spaces.Box(
        #         low=np.array([0, 0, -np.inf, -np.inf, 0, 0]),
        #         high=np.array([screen_size, screen_size, np.inf, np.inf, screen_size, screen_size]),
        #         dtype=np.float32
        #     )
        # elif state_mode == 'image':
        #     self.observation_space = spaces.Box(
        #         low=0,
        #         high=255,
        #         shape=(int(screen_size), int(screen_size), 3),
        #         dtype=np.uint8
        #     )

        # Initialize state variables
        self.agent_pos = None
        self.agent_vel = None
        self.goal_pos = None
        self.fig, self.ax = None, None  # For graphical rendering
        self.render_size = render_size

    def reset(self, seed=42, options={}, return_info=False):
        """
        Resets the environment to an initial state.
        """
        self.agent_pos = self._get_random_position()
        self.agent_vel = np.zeros(2, dtype=np.float32)
        self.goal_pos = self._get_random_position()

        # Ensure agent and goal do not overlap
        while np.array_equal(self.goal_pos, self.agent_pos):
            self.goal_pos = self._get_random_position()

        return self._get_observation()

    def step(self, action):
        """
        Takes a step in the environment.
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Update velocity and position
        self.agent_vel += action
        self.agent_pos += self.agent_vel

        # Clamp position to screen bounds
        self.agent_pos = np.clip(self.agent_pos, 0, self.screen_size)

        reward = -1  # Penalize for each step

        # Check for collisions with obstacles
        for obs in self.obstacles:
            if np.linalg.norm(self.agent_pos - np.array(obs)) < 0.5:  # Collision threshold
                reward -= 10

        # Check if goal is reached
        done = np.linalg.norm(self.agent_pos - self.goal_pos) < 0.5  # Goal threshold
        if done:
            reward += 10  # Reward for reaching the goal

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        """
        Renders the environment.
        """
        if mode == 'human':
            self._render_window()
        elif mode == 'rgb_array':
            return self._render_image()

    def close(self):
        """
        Clean up resources (if any).
        """
        if self.fig is not None:
            plt.close(self.fig)

    def _get_observation(self):
        """
        Returns the current observation.
        """
        if self.state_mode == 'ground_truth':
            return {'state': np.concatenate([self.agent_pos, self.agent_vel, self.goal_pos])}
        elif self.state_mode == 'image':
            return self._render_image()

    def _get_random_position(self):
        """
        Returns a random position on the screen that is not an obstacle.
        """
        while True:
            pos = self.np_random.uniform(0, self.screen_size, size=(2,))
            if all(np.linalg.norm(pos - np.array(obs)) >= 0.5 for obs in self.obstacles):
                return pos

    def _render_image(self):
        """
        Renders the environment as an RGB array.
        """
        image = np.zeros((int(self.screen_size), int(self.screen_size), 3), dtype=np.uint8)
        for y, x in self.obstacles:
            image[int(y), int(x)] = [255, 0, 0]  # Red for obstacles
        if int(self.goal_pos[1]) < self.screen_size and int(self.goal_pos[0]) < self.screen_size:
            image[int(self.goal_pos[1]), int(self.goal_pos[0])] = [0, 255, 0]  # Green for the goal
        if int(self.agent_pos[1]) < self.screen_size and int(self.agent_pos[0]) < self.screen_size:
            image[int(self.agent_pos[1]), int(self.agent_pos[0])] = [0, 0, 255]  # Blue for the agent
        image_resized = cv2.resize(image, (self.render_size, self.render_size), interpolation=cv2.INTER_NEAREST)
        return image_resized

    def _render_window(self):
        """
        Renders the environment in a matplotlib window.
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.ax.set_xlim(-0.5, self.screen_size - 0.5)
            self.ax.set_ylim(-0.5, self.screen_size - 0.5)
            self.ax.set_xticks(np.arange(0, self.screen_size + 1, 1))
            self.ax.set_yticks(np.arange(0, self.screen_size + 1, 1))
            self.ax.grid(True)

        self.ax.clear()
        self.ax.set_xlim(-0.5, self.screen_size - 0.5)
        self.ax.set_ylim(-0.5, self.screen_size - 0.5)
        self.ax.grid(True)

        for y, x in self.obstacles:
            self.ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='red'))

        self.ax.add_patch(plt.Circle((self.goal_pos[0], self.goal_pos[1]), 0.3, color='green'))
        self.ax.add_patch(plt.Circle((self.agent_pos[0], self.agent_pos[1]), 0.3, color='blue'))

        plt.pause(0.1)