import torch
import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, n_hidden, output_dim, device):
        super(SimpleMLP, self).__init__()
        self.in_dim = input_dim
        self.n_hidden = n_hidden
        self.out_dim = output_dim
        self.gpu_device = device
        self.seq = nn.Sequential(
            nn.Linear(input_dim, n_hidden, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, 2 * output_dim, device=device),
            NormalParamExtractor(),
        )

    def forward(self, x):        
        # Check if the input tensor has the right number of sequences
        if x.shape[-1] != self.in_dim:
            # Add zeros to the tensor where x has size (batch_size, seq_len, state_dim)
            x = torch.cat([x, torch.zeros(self.in_dim - x.shape[-1]).to(self.gpu_device)], dim=-1)

        loc, scale = self.seq(x)

        return loc, scale