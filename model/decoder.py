import torch
import torch.nn as nn
from model.attention import Attention

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        # input: (batch_size)
        # last_hidden: (decoder.n_layers, batch_size, hidden_size)
        # encoder_outputs: (timestep, batch_size, hidden_size)
        embedded = self.embed(input).unsqueeze(0)
        # embedded: (1, batch_size, target_embed_size)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        # last_hidden[-1]: (batch_size, hidden_size)
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        # att_weights: (batch_size, 1, timestep)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # context: (batch_size, 1, hidden_size)
        context = context.transpose(0, 1)
        # context: (1, batch_size, hidden_size)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], dim=2)
        # rnn_input: (1, batch_size, target_embed_size + hidden_size)
        output, hidden = self.gru(rnn_input, last_hidden)
        # output: (1, batch_size, hidden_size)
        # hidden: (decoder.n_layers, batch_size, hidden_size)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        # output: (batch_size, hidden_size)
        context = context.squeeze(0)
        # context: (batch_size, hidden_size)
        output = self.out(torch.cat([output, context], dim=1))
        # output: (batch_size, target_vocab_size)
        # output = torch.log_softmax(output, dim=1)
        return output, hidden