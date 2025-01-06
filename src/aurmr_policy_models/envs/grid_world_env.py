import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt


def convert_to_int_if_ndarray(obj):
    """Convert a numpy ndarray to an int if it is a single-element array."""
    if isinstance(obj, np.ndarray):
        if obj.size == 1:  # Ensure it's a single-element array
            return int(obj)
        else:
            raise ValueError("Cannot convert multi-element ndarray to int.")
    return obj


class GridWorldEnv(gym.Env):
    """
    A configurable 2D Grid World Environment for Reinforcement Learning.
    """

    def __init__(self, env_name='grid_world', grid_size=5, blocked_cells=None, render=False, state_mode='ground_truth'):
        super(GridWorldEnv, self).__init__()
        
        self.env_name = env_name
        self.grid_size = grid_size
        self.blocked_cells = blocked_cells if blocked_cells else []
        self.render_enabled = render
        self.state_mode = state_mode  # 'ground_truth' or 'image'

        # Define action space: 4 actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Define observation space
        if state_mode == 'ground_truth':
            self.observation_space = spaces.Dict({
                'agent_pos': spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
                'goal_pos': spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
                'grid': spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.int32)
            })
        elif state_mode == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(grid_size, grid_size, 3), dtype=np.uint8)

        # Initialize positions
        self.agent_pos = None
        self.goal_pos = None
        self.fig, self.ax = None, None  # For graphical rendering

    def reset(self):
        """
        Resets the environment to an initial state.
        """
        # Set random positions for the agent and the goal
        self.agent_pos = self._get_random_position()
        self.goal_pos = self._get_random_position()

        # Ensure agent and goal do not overlap
        while tuple(self.goal_pos) == tuple(self.agent_pos):
            self.goal_pos = self._get_random_position()

        return self._get_observation()

    def step(self, action):
        """
        Takes a step in the environment.
        """
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        move_map = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

        # import pdb; pdb.set_trace()
        action = convert_to_int_if_ndarray(action)
        new_pos = self.agent_pos + np.array(move_map[action])
        reward = -1  # Penalize for each step

        # Check for walls and boundaries
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        else:
            reward -= 5  # Penalize for invalid move
        
        is_blocked = tuple(self.agent_pos) in self.blocked_cells
        if is_blocked:
            reward -= 10

        # Check if goal is reached
        done = np.array_equal(self.agent_pos, self.goal_pos)
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
            # grid = self._get_grid()
            return np.concatenate([self.agent_pos, self.goal_pos])
            # return {
            #     'agent_pos': self.agent_pos.copy(),
            #     'goal_pos': self.goal_pos.copy(),
            #     'grid': grid
            # }
        elif self.state_mode == 'image':
            return self._render_image()

    def _get_grid(self):
        """
        Generates a grid representation of the environment.
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for cell in self.blocked_cells:
            grid[cell] = 1  # Blocked cell
        return grid

    def _get_random_position(self):
        """
        Returns a random position on the grid that is not blocked.
        """
        while True:
            pos = np.random.randint(0, self.grid_size, size=(2,))
            if tuple(pos) not in self.blocked_cells:
                return pos

    def _is_valid_position(self, pos):
        """
        Checks if a position is valid (within bounds and not blocked).
        """
        return (0 <= pos[0] < self.grid_size and
                0 <= pos[1] < self.grid_size)
                # tuple(pos) not in self.blocked_cells)

    def _render_image(self):
        """
        Renders the environment as an RGB array.
        """
        image = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for y, x in self.blocked_cells:
            image[y, x] = [255, 0, 0]  # Red for blocked cells
        image[self.goal_pos[0], self.goal_pos[1]] = [0, 255, 0]  # Green for the goal
        image[self.agent_pos[0], self.agent_pos[1]] = [0, 0, 255]  # Blue for the agent
        return image

    def _render_window(self):
        """
        Renders the grid world in a matplotlib window.
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.ax.set_xlim(-0.5, self.grid_size - 0.5)
            self.ax.set_ylim(-0.5, self.grid_size - 0.5)
            self.ax.set_xticks(range(self.grid_size))
            self.ax.set_yticks(range(self.grid_size))
            self.ax.grid(True)

        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.grid(True)

        for y, x in self.blocked_cells:
            self.ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='red'))

        self.ax.add_patch(plt.Circle((self.goal_pos[1], self.goal_pos[0]), 0.3, color='green'))
        self.ax.add_patch(plt.Circle((self.agent_pos[1], self.agent_pos[0]), 0.3, color='blue'))

        plt.pause(0.1)
