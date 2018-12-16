import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (src_timestep, batch_size)
        # trg: (trg_timestep, batch_size)
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, vocab_size)
        encoder_output, hidden = self.encoder(src)
        # encoder_output: (timestep, batch_size, hidden_size)
        # hidden: (encoder.n_layers * 2, batch_size, hidden_size)
        hidden = hidden[:self.decoder.n_layers]
        # hidden: (decoder.n_layers, batch_size, hidden_size)
        output = trg[0, :]  # sos
        for t in range(1, max_len):
            output, hidden = self.decoder(
                    output, hidden, encoder_output)
            # output: (batch_size, target_vocab_size)
            # hidden: (decoder.n_layers, batch_size, hidden_size)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = trg[t] if is_teacher else top1
        return outputs