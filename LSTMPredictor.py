import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, q_size, n_experiments, dropout_prob=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_size + q_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc3 = nn.Linear(n_experiments*hidden_size//4, hidden_size//4)
        self.fc4 = nn.Linear(hidden_size//4, output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, xs, qs):
        B = xs.shape[0]
        E = xs.shape[1]
        experiment_outs = []
        for i in range(E):
            x = xs[:, i]
            q = qs[:, i]
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = F.relu(self.fc1(torch.cat([out[:,-1,:], q], dim=1)))
            out = F.relu(self.fc2(out))
            experiment_outs.append(out)
        out = torch.cat(experiment_outs, dim=1)
        out = self.fc3(out.view(B, -1))
        out = self.dropout(out)
        out = self.fc4(out)
        return out
