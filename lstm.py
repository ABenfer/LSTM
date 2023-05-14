import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import datetime
import config
import importlib
import os
import json

importlib.reload(config)

print("ver lstm = 0.4.16")
# print all variables from config
for key, value in config.__dict__.items():
    if not key.startswith("__"):
        print(key, value)

now = datetime.datetime.now()
date = now.strftime("%Y_%m_%d")
time = now.strftime("%H_%M_%S")


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, q_size, n_experiments):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size + q_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc3 = nn.Linear(n_experiments*hidden_size//4, output_size)

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
            out = self.fc1(torch.cat([out[:,-1,:], q], dim=1))
            out = self.fc2(out)
            experiment_outs.append(out)
        out = torch.cat(experiment_outs, dim=1)
        out = self.fc3(out.view(B, -1))
        return out


class JsonDataset(Dataset):
    def __init__(self, json_files):
        # json_files is a list of file paths
        self.json_files = json_files
        print("len json_files: ", len(self.json_files))
        self.data = []
        for file in self.json_files:
            with open(file, 'r') as f:
                self.data.append(json.load(f))

        print("len(self.data): ", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        k = torch.tensor(item['k']).float()
        d = torch.tensor(item['d']).float()
        kd = torch.cat((k, d), 0)

        experiments_ts, experiments_q = [], []
        for experiment in item['experiments']:
            time = torch.tensor(experiment['data']['time']).float()
            fx = torch.tensor(experiment['data']['fx']).float()
            fy = torch.tensor(experiment['data']['fy']).float()
            fz = torch.tensor(experiment['data']['fz']).float()
            x = torch.tensor(experiment['data']['x']).float()
            y = torch.tensor(experiment['data']['y']).float()
            z = torch.tensor(experiment['data']['z']).float()

            q = torch.tensor(experiment['q']).float()
            experiments_ts.append((time, fx, fy, fz, x, y, z))
            experiments_q.append(q)

        return experiments_ts, experiments_q, kd



def normalize(samples_ts_flattened, samples_q_flattened, y_flattened):
    # samples_ts_flattened
    mean_ts = samples_ts_flattened.mean(dim=0, keepdim=True)  # Compute the mean along dimension 'x'
    std_ts = samples_ts_flattened.std(dim=0, keepdim=True)  # Compute the standard deviation along dimension 'x'

    # Normalize the tensor along dimension 'x'
    normalized_tensor = (samples_ts_flattened - mean_ts) / (std_ts + 1e-7)  # adding a small constant to prevent division by zero

    # samples_q_flattened
    mean_q = samples_q_flattened.mean(dim=0, keepdim=True)
    std_q = samples_q_flattened.std(dim=0, keepdim=True)

    normalized_q = (samples_q_flattened - mean_q) / (std_q + 1e-7)

    # y_flattened
    mean_y = y_flattened.mean(dim=0, keepdim=True)
    std_y = y_flattened.std(dim=0, keepdim=True)

    normalized_y = (y_flattened - mean_y) / (std_y + 1e-7)

    # store all scalers in a list
    norm_parameter = {"mean_ts": mean_ts, "std_ts": std_ts, 
                      "mean_q": mean_q, "std_q": std_q, 
                      "mean_y": mean_y, "std_y": std_y}

    return normalized_tensor, normalized_q, normalized_y, norm_parameter


def inverse_normilize(normalized_y, norm_parameter):
    mean_y = norm_parameter["mean_y"]
    std_y = norm_parameter["std_y"]

    y = normalized_y * std_y + mean_y

    return y


def load_and_normalize_data(json_files):
    dataset = JsonDataset(json_files)
    samples_ts, samples_q, y = zip(*dataset)

    # Flatten all data for rescaling, sample_ts has shape [S, E, C, T]
    # flatten everything except channels so final dimension is [S*E*T, C]
    samples_ts_stacked = torch.stack([torch.stack([torch.stack(sub_sub_list) for sub_sub_list in sub_list]) for sub_list in samples_ts])
    samples_ts_flattened = samples_ts_stacked.transpose(2,3).flatten(end_dim=2)

    # sample_q has shape [S, E, Q]
    # flatten everything so final dimension is [S*E, Q]
    samples_q_stacked = torch.stack([torch.stack(sub_list) for sub_list in samples_q])
    samples_q_flattened = samples_q_stacked.flatten(end_dim=1)

    y_flattened = torch.stack(y)

    # Rescale the data
    ts_data_norm, q_data_norm, y_data_norm, norm_parameter = normalize(samples_ts_flattened, samples_q_flattened, y_flattened)

    # Split the rescaled data back into experiments
    # ts_data_scaled has shape [S*E*T, C]
    # split into [S, E, T, C]
    S, E, T, C = samples_ts_stacked.shape
    samples_ts_norm = ts_data_norm.view(S, -1, C, T).transpose(2,3)

    # q_data_scaled has shape [S*E, Q]
    # split into [S, E, Q]
    Q = samples_q_stacked.shape[2]
    samples_q_norm = q_data_norm.view(S, -1, Q)

    return samples_ts_norm, samples_q_norm, y_data_norm, norm_parameter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load and rescale data
json_files = np.sort([config.SIMS_PATH + el for el in os.listdir(config.SIMS_PATH) if el.endswith('.json')])[:]
samples_ts, samples_q, y, norm_parameter = load_and_normalize_data(json_files)

samples_ts = np.swapaxes(samples_ts, 2, 3) #[S, E, C, T] -> [S, E, T, C]

print("samples_ts.shape: ", np.array(samples_ts).shape) # samples_ts: [S, E, T, C]
print("samples_q.shape: ", np.array(samples_q).shape) # samples_q: [S, E, Q]
print("y.shape: ", np.array(y).shape) # y: [S, O]

S = np.array(samples_ts).shape[0]
E = np.array(samples_ts).shape[1]
T = np.array(samples_ts).shape[2]
C = np.array(samples_ts).shape[3]
Q = np.array(samples_q).shape[2]
O = np.array(y).shape[1]


# Split data into training and validation sets
test_size, val_size = 0.05, 0.1
test_indices = list(np.random.choice(S, size=int(test_size*S), replace=False))
remaining_indices = list(set(range(S)) - set(test_indices))
val_indices = list(np.random.choice(remaining_indices, size=int(val_size*S), replace=False))
train_indices = list(set(remaining_indices) - set(val_indices))

train_dataset = [(samples_ts[i], samples_q[i], y[i]) for i in train_indices]
val_dataset = [(samples_ts[i], samples_q[i], y[i]) for i in val_indices]

# Create train and validation data loaders
batch_size = config.BATCH_SIZE
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



hidden_size = 128
num_layers = 2
model = LSTMPredictor(input_size=C, hidden_size=128, num_layers=2, output_size=O, q_size=Q, n_experiments=E).to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.LEARNING_RATE_DECAY)






num_epochs = config.EPOCHS
losses, val_losses, best_val_loss = [], [], 99999
start_time = datetime.datetime.now()
for epoch in range(num_epochs):
    # train the model
    train_loss = 0
    for i, (xs, qs, y) in enumerate(train_loader):
        # Forward pass
        xs = xs.to(device)
        qs = qs.to(device)
        y = y.to(device)
        outputs = model(xs, qs)
        loss = criterion(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimization step
        optimizer.step()
        train_loss += loss.item()
    losses.append(train_loss / len(train_loader))

    # validate the model
    val_loss = 0
    for i, (xs, qs, y) in enumerate(val_loader):
        xs = xs.to(device)
        qs = qs.to(device)
        y = y.to(device)
        outputs = model(xs, qs)
        loss = criterion(outputs, y)
        val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

    # print the progress including the time and the etf and * if a new record is set
    estimated_time_finished = start_time + (datetime.datetime.now() - start_time) / (epoch + 1) * num_epochs

    epoch_num, total_epochs = epoch + 1, num_epochs
    train_loss, val_loss = losses[-1], val_losses[-1]
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    finished_time = estimated_time_finished.strftime("%H:%M:%S")
    best_loss_indicator = '*' if val_loss < best_val_loss else ''
    
    print(f"Epoch [{epoch_num}/{total_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {current_time}, Finished: {finished_time} {best_loss_indicator}")

    # new best validation loss if the validation loss has decreased
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        if epoch > 20:
            pass
            # torch.save(model.state_dict(), config.CHECKPOINT_PATH + "model_{}_{}.pth".format(date, time))

    # break if the validation loss hasn't improved for some epochs
    if epoch > config.EARLY_STOPPING and not best_val_loss in val_losses[-config.EARLY_STOPPING:]:
        print("Early stopping")
        break

    
# print an example prediction
with torch.no_grad():
    xs, qs, y = next(iter(val_loader))
    xs = xs.to(device)
    qs = qs.to(device)
    y = y.to(device)
    norm_parameter = {key: value.to(device) for key, value in norm_parameter.items()}
    outputs = model(xs, qs)
    outputs = inverse_normilize(outputs, norm_parameter)
    y = inverse_normilize(y, norm_parameter)
    for estimation, real in zip(outputs, y):
        for el_e, el_r in zip(estimation, real):
            el_e = el_e.item()
            el_r = el_r.item()
            print("estimation: {} - real: {} - rel. error: {}%".format(round(el_e,1), round(el_r,1), round(100*(el_e - el_r) / el_r),1))
        print("------")
        


# plot the training and validation loss
plt.plot(losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Samples: {}, Data Points: {}, Batch size: {}, Learning rate: {}"
          .format(S, T, config.BATCH_SIZE, config.LEARNING_RATE))
plt.grid()
plt.ylim(bottom=0)
plt.legend()
plt.savefig(config.PLOT_PATH + "model_{}_{}.png".format(date, time))



# x_time_series = torch.tensor(x_time_series).float().to(device)
# x_real_features = torch.tensor(x_real_features).float().to(device)
# y_real_features = torch.tensor(y_real_features).float().to(device)


# # split the data into train and validation sets
# test_size, val_size = 0.05, 0.1
# num_samples = len(x_time_series)
# test_indices = list(np.random.choice(num_samples, size=int(test_size*num_samples), replace=False))
# remaining_indices = list(set(range(num_samples)) - set(test_indices))
# val_indices = list(np.random.choice(remaining_indices, size=int(val_size*num_samples), replace=False))
# train_indices = list(set(remaining_indices) - set(val_indices))

# train_dataset = TensorDataset(x_time_series[train_indices], x_real_features[train_indices], y_real_features[train_indices])
# val_dataset = TensorDataset(x_time_series[val_indices], x_real_features[val_indices], y_real_features[val_indices])

# usage
# json_files = np.sort([el for el in os.listdir(config.SIMS_INTERP_PATH) if el.endswith('.json')])[:config.N_DATA_POINTS]
# dataset = JsonDataset(json_files)
# loader = DataLoader(dataset, batch_size=10, shuffle=True)

# split the data into train and validation sets
# rescale data
# maybe swap axes


# create train and validation data loaders
# batch_size = config.BATCH_SIZE
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# # val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)