import numpy as np
from aurmr_policy_models.agents.base_agent import BaseAgent


class GridWorldModelAgent(BaseAgent):
    def __init__(self, env, agent_name, version, model, device="cuda:0"):
        super().__init__(env, agent_name, version)
        self.model = model.to(device)
        self.device = device

    def select_action(self, observation):
        # Convert observation to a tensor and process it through the model
        state_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        conditions = {"state": state_tensor.unsqueeze(1)}  # Add batch and sequence dimension
        with torch.no_grad():
            action = self.model(conditions)
        return action.squeeze(0).cpu().numpy()

if __name__ == "__main__":
    print("MLPModel and MLPAgent are ready to be used.")