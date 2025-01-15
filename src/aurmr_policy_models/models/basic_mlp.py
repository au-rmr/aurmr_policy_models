import torch
import torch.nn as nn

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, model_name, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.model_name = model_name
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.is_discrete = False

    def forward(self, cond):
        print("MLP?")
        # Extract the last state from the conditions
        # x = x[:, -1, :]  # Assumes 'state' contains the relevant state information
        return self.model(cond['state']).unsqueeze(0)
    
    def loss(self, actions, cond):
        predicted_actions = self.forward(cond)
        if self.is_discrete:
            # Apply Cross-Entropy Loss for discrete prediction
            return torch.nn.functional.cross_entropy(predicted_actions.view(-1, predicted_actions.size(-1)), actions.view(-1).long())
        else:
            # Apply MSE Loss for continuous prediction
            # import pdb; pdb.set_trace()
            return torch.nn.functional.mse_loss(predicted_actions, actions.squeeze(-1))

