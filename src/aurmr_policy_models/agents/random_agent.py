import numpy as np

from aurmr_policy_models.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    An agent that selects actions randomly from the available action space.
    """
    
    def select_action(self, observation):
        """
        Selects a random action from the environment's action space.
        
        Parameters:
            observation (np.array): The current observation from the environment (not used here).
        
        Returns:
            action (np.array): A randomly chosen action.
        """
        return self.env.action_space.sample()
