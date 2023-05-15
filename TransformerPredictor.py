import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    # def __init__(self, input_size, hidden_size, num_layers, output_size, q_size, n_experiments, dropout_prob=0):

    def __init__(self, input_dim, output_dim, nhead, num_layers, sequen_len, q_size, dropout=0):
        super(TransformerModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, nhead, dim_feedforward=2048, dropout=dropout),
            num_layers)
        self.fc1 = nn.Linear(input_dim * sequen_len + q_size, input_dim * sequen_len//2)
        self.fc2 = nn.Linear(input_dim * sequen_len//2, input_dim * sequen_len//10)
        self.fc3 = nn.Linear(input_dim * sequen_len//10, output_dim)

    def forward(self, src, q):
        src = src.permute(1, 0, 2)  # shape (seq_len, batch_size, input_dim)
        out = self.transformer_encoder(src)
        out = out.permute(1, 0, 2)  # shape (batch_size, seq_len, hidden_dim)
        out = out.reshape(out.size(0), -1)  # flatten the tensor
        out = self.fc1(torch.concat([out, q], dim=1))
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    