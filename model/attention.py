import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    # additive attention
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_size)
        # encoder_outputs: (timestep, batch_size, hidden_size)
        timestep = encoder_outputs.size(0)
        # timestep: T
        # hidden.repeat(timestep, 1, 1): (timestep, batch_size, hidden_size)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        # h: (batch_size, timestep, hidden_size)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        # encoder_outputs: (batch_size, timestep, hidden_size)
        attn_energies = self.score(h, encoder_outputs)
        # attn_energies: (batch_size, timestep)
        # return F.relu(attn_energies).unsqueeze(1) # (batch_size, 1, timestep)
        return torch.softmax(attn_energies, dim=1).unsqueeze(1) # (batch_size, 1, timestep)

    def score(self, hidden, encoder_outputs):
        # hidden: (batch_size, timestep, hidden_size)
        # encoder_outputs: (batch_size, timestep, hidden_size)
        # [hidden; encoder_outputs]: (batch_size, timestep, hidden_size * 2)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        # energy: (batch_size, timestep, hidden_size)
        energy = energy.transpose(1, 2)
        # energy: (batch_size, hidden_size, timestep)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        # v: (batch_size, 1, hidden_size)
        energy = torch.bmm(v, energy)
        # energy: (batch_size, 1, timestep)
        return energy.squeeze(1)  # (batch_size, timestep)