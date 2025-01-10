import numpy as np
from aurmr_policy_models.agents.base_agent import BaseAgent

class PointMassExpertPlannerAgent(BaseAgent):
    """
    An expert agent for the PointMassEnv that accelerates toward the goal.
    """

    def select_action(self, observation):
        """
        Selects the acceleration action to move the agent closer to the goal.

        Parameters:
            observation (np.array): The current observation consisting of:
                [agent_pos_x, agent_pos_y, agent_vel_x, agent_vel_y, goal_pos_x, goal_pos_y]

        Returns:
            action (np.array): The acceleration to apply (ax, ay).
        """
        agent_pos = observation['state'][:2]  # Extract agent position (x, y)
        agent_vel = observation['state'][2:4]  # Extract agent velocity (vx, vy)
        goal_pos = observation['state'][4:6]  # Extract goal position (x, y)

        # Compute the direction vector toward the goal
        direction = goal_pos - agent_pos

        # Normalize the direction vector to unit length
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm

        # Compute desired velocity toward the goal
        desired_velocity = direction * 1.0  # Set a reasonable target speed

        # Compute the acceleration needed to achieve the desired velocity
        acceleration = desired_velocity - agent_vel

        # Clip acceleration to within the allowed range of the environment
        acceleration = np.clip(acceleration, -1.0, 1.0)

        return [acceleration]
