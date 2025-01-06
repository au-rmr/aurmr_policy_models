import numpy as np
from aurmr_policy_models.agents.base_agent import BaseAgent

class GridWorldExpertPlannerAgent(BaseAgent):
    """
    An expert agent that always takes the shortest path to the goal in a grid world.
    """

    def select_action(self, observation):
        """
        Selects the action that moves the agent closer to the goal.

        Parameters:
            observation (np.array): The current observation (not used in this implementation).

        Returns:
            action (int): The action to take (0=up, 1=down, 2=left, 3=right).
        """
        agent_pos = self.env.agent_pos  # Get agent position from the environment
        goal_pos = self.env.goal_pos    # Get goal position from the environment

        # Compute the difference between agent and goal positions
        delta = goal_pos - agent_pos

        # Determine the best action to minimize Manhattan distance to the goal
        if abs(delta[0]) > abs(delta[1]):
            # Prioritize vertical movement
            if delta[0] > 0:
                return 1  # Down
            else:
                return 0  # Up
        else:
            # Prioritize horizontal movement
            if delta[1] > 0:
                return 3  # Right
            else:
                return 2  # Left
