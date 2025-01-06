import torch
import torch.nn as nn

# Define the RNN Model
class RNN(nn.Module):
    def __init__(self, model_name, input_dim, output_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.model_name = model_name
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Process the sequence of actions through the RNN
        # x shape: (Batch, Sequence Length, Action Size)
        rnn_out, _ = self.rnn(x)  # rnn_out shape: (Batch, Sequence Length, Hidden Size)
        last_hidden_state = rnn_out[:, -1, :]  # Extract the last hidden state (Batch, Hidden Size)
        return self.fc(last_hidden_state)  # Output shape: (Batch, Output Dim)