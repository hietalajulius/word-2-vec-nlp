import torch
import torch.nn as nn

class GRU(nn.Module):
    """

    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 output_dim, n_layers, bidirectional, dropout, drop_out_dense=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(drop_out_dense)

    def forward(self, text):
        embedded_text = self.dropout(self.embedding(text))
        output, hidden = self.gru(embedded_text)
        hidden = self.dropout2(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        x = self.fc(hidden.squeeze(0))
        return x