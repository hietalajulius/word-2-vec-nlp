import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class RNNModel(nn.Module):
    """

    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 output_dim, n_layers, bidirectional, dropout, pad_idx, use_gru=True):
        super().__init__()
        self.n_hidden = hidden_dim
        self.n_layers = n_layers
        self.use_gru = use_gru
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if use_gru:
            self.rnn = nn.GRU(embedding_dim,
                              hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              dropout=0 if n_layers < 2 else dropout,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(embedding_dim,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               dropout=0 if n_layers < 2 else dropout,
                               batch_first=True)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded_text = self.dropout(self.embedding(text))
        # print(f"embedded_text {embedded_text.shape}")
        packed_embedded = pack_padded_sequence(embedded_text, text_lengths, batch_first=True)
        packed_output, hidden = self.rnn(packed_embedded)
        # output, output_lengths = pad_packed_sequence(packed_output)

        if self.use_gru:
            x = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hn, cn = hidden
            x = self.dropout(torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1))
        #print(f"x.shape {x.shape}")
        x = self.fc(x.squeeze(0))
        #print(f"x.shape {x.shape} {x}")
        # x = F.log_softmax(output, dim=-1)
        # x = x.squeeze(1)
        # print(f"x.shape {x.shape}")
        return x
