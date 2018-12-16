import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5): 
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        # src: (timestep, batch_size)
        embedded = self.embed(src)
        # embedded: (timestep, batch_size, embed_size)
        outputs, hidden = self.gru(embedded, hidden)
        # outputs: (timestep, batch_size, hidden_size * 2) bidirectional=True
        # hidden: (n_layers * 2, batch_size, hidden_size) bidirectional=True
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        # outputs: (timestep, batch_size, hidden_size)
        return outputs, hidden