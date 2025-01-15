import os
import torch
import numpy as np
from omegaconf import OmegaConf
from aurmr_policy_models.agents.base_agent import BaseAgent
from aurmr_policy_models.utils.config_utils import load_config_from_file, instantiate

class LearnedAgent(BaseAgent):
    """
    An agent that uses a trained model to select actions in an environment.
    Extends the BaseAgent class.
    """
    
    def __init__(self, env, agent_name, version, model, device="cuda:0", deterministic=True):
        super().__init__(env, agent_name, version)
        self.device = device

        # self.config = load_config_from_file(os.path.join(model_path, config_name))

        # self.model = instantiate(self.config.model)
        # print("learned agent model:")
        # print(self.model)
        self.model = model
        # self.model.load_state_dict(torch.load(os.path.join(model_path, model_name))['model'])
        self.model.eval()  # Set model to evaluation mode
        self.model.to(self.device)

        self.deterministic = deterministic
    
    def preprocess_observation(self, observation):
        """
        Preprocess the observation to match the input format expected by the model.

        Parameters:
            observation (np.array): The current observation from the environment.
        
        Returns:
            torch.Tensor: The preprocessed observation.
        """
        # Convert observation to torch tensor and send to device
        obs_tensor = torch.tensor(observation['state'], dtype=torch.float32).to(self.device)
        # Add batch dimension if required by the model
        return {'state': obs_tensor.unsqueeze(0)}  # Shape: (1, ..., ...) if needed

    def postprocess_action(self, sample):
        """
        Convert the model's output tensor into a format suitable for the environment.

        Parameters:
            sample The raw sample output by the model.
        
        Returns:
            np.array: The processed action.
        """
        # if self.config.model.is_discrete:
        #     # Apply softmax to convert logits to probabilities
        #     probabilities = torch.softmax(sample.trajectories, dim=-1)
        #     # Sample an action based on the probabilities
        #     action_index = torch.multinomial(probabilities, num_samples=1).item()
        #     return action_index
        # else:
        # Remove batch dimension and convert to numpy for continuous actions
        return sample.trajectories.squeeze(0).cpu().detach().numpy()
        # return sample.squeeze(0).cpu().detach().numpy()
    
    def select_action(self, observation):
        """
        Select an action based on the current observation using the trained model.

        Parameters:
            observation (np.array): The current observation from the environment.
        
        Returns:
            action (np.array): The action chosen by the model.
        """
        # Preprocess the observation
        obs_tensor = self.preprocess_observation(observation)
        
        # Get the model's output
        with torch.no_grad():
            sample = self.model(obs_tensor, deterministic=self.deterministic)
        
        # Postprocess the model's output to obtain the action
        return self.postprocess_action(sample)
