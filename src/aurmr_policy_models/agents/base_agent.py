from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    A generic base agent class that can interact with any OpenAI gym environment.
    This class uses abc module to enforce that subclasses implement the select_action method.
    """
    
    def __init__(self, env, agent_name, version):
        self.env = env
        self.agent_name = agent_name
        self.version = version
    
    @abstractmethod
    def select_action(self, observation):
        """
        Selects an action based on the current observation from the environment.
        
        This abstract method must be implemented by any subclass.
        
        Parameters:
            observation (np.array): The current observation from the environment.
        
        Returns:
            action (np.array): The action chosen by the agent.
        """
        pass
    
    def run_episode(self):
        """
        Runs a single episode with the environment using the agent's action selection policy.
        
        Returns:
            total_reward (float): The total reward obtained in the episode.
        """
        observation = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = self.select_action(observation)
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
        
        return total_reward
