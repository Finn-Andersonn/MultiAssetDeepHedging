import torch.nn as nn

class RiskAverseCritic(nn.Module):
    """
    Approximates V(s). We'll do a simple feedforward net:
    input = (z,m) flattened => hidden => output a single scalar V(s).
    """
    def __init__(self, state_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        """
        state: shape [batch_size, state_dim]
        returns shape [batch_size], the value function
        """
        x = self.net(state)
        return x.squeeze(-1)