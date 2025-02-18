# models/policy_network.py

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

class GaussianPolicyActor(nn.Module):
    """
    An LSTM-based actor that outputs *two* heads:
      - mean in R^action_dim
      - log_std in R^action_dim (can be trainable or partially fixed)
    We interpret the action as a normal with mean=mean_head, std=exp(log_std_head).

    We'll do advantage-based PG:
       actor_loss = - advantage * log_prob(a|s)
    """
    def __init__(self, state_dim, action_dim, hidden_size=32):
        super().__init__()
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.hidden_size = hidden_size

        self.lstm = nn.LSTMCell(state_dim, hidden_size)
        # we'll define separate linear heads for mean and log_std
        self.fc_mean   = nn.Linear(hidden_size, action_dim)
        self.fc_logstd = nn.Linear(hidden_size, action_dim)

    def forward(self, state, hx, cx):
        """
        state: shape [batch_size, state_dim]
        hx,cx: shape [batch_size, hidden_size]
        returns (mean, log_std, hx_next, cx_next)
        """
        hx_next, cx_next = self.lstm(state, (hx, cx))
        mean   = self.fc_mean(hx_next)
        logstd = self.fc_logstd(hx_next)
        return mean, logstd, hx_next, cx_next

    def init_hidden(self, batch_size=1):
        hx = torch.zeros(batch_size, self.hidden_size)
        cx = torch.zeros(batch_size, self.hidden_size)
        return hx, cx


def gaussian_log_prob(actions, mean, log_std):
    """
    actions, mean, log_std: shape [batch_size, action_dim]
    We compute the log probability of each dimension's normal distribution
    and sum across action_dim => shape [batch_size].
    normal dist = (1/sqrt(2pi*var)) * exp(-0.5*(x-mean)^2 / var)
    log_prob_i = -( (x-mean)^2 / (2 * exp(2*log_std)) + log_std + 0.5*log(2*pi) )
    sum across dimension => total log_prob
    """
    var     = torch.exp(2.0*log_std)
    diff    = (actions-mean)**2
    logprob = -0.5*(diff/var + math.log(2.0*math.pi)) - log_std
    return logprob.sum(dim=1) # sum across action_dim
