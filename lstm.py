import torch
import torch.nn as nn
import numpy as np
import data_import
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from matplotlib import pyplot as plt
import datetime
import config


print("ver lstm = 0.3.4")
# print all variables from config
for key, value in config.__dict__.items():
    if not key.startswith("__"):
        print(key, value)

now = datetime.datetime.now()
date = now.strftime("%Y_%m_%d")
time = now.strftime("%H_%M_%S")


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size + x_real_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc3 = nn.Linear(hidden_size//4, output_size)

    def forward(self, x, q):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(torch.concat([out[:,-1,:], q], dim=1))
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    

def rescale(x_time_series, x_real_features, y_real_features):
    # Reshape the input data to 2D arrays
    ts_data = x_time_series.reshape(x_time_series.shape[0], -1)
    x_real_data = x_real_features.reshape(x_real_features.shape[0], -1)
    y_real_data = y_real_features.reshape(y_real_features.shape[0], -1)

    # Create a MinMaxScaler object and fit it to the time series data
    ts_scaler = MinMaxScaler()
    ts_scaler.fit(ts_data)

    # Create a separate MinMaxScaler object and fit it to the real data
    x_real_scaler = MinMaxScaler()
    x_real_scaler.fit(x_real_data)

    # Create a separate MinMaxScaler object and fit it to the real data
    y_real_scaler = MinMaxScaler()
    y_real_scaler.fit(y_real_data)

    # Scale the time series data
    ts_data_scaled = ts_scaler.transform(ts_data).reshape(x_time_series.shape)
    # Scale the real data using the same scaler
    x_real_data_scaled = x_real_scaler.transform(x_real_data).reshape(x_real_features.shape)
    y_real_data_scaled = y_real_scaler.transform(y_real_data).reshape(y_real_features.shape)

    # Reshape the scaled data back to the original shape

    return ts_data_scaled, x_real_data_scaled, y_real_data_scaled, y_real_scaler


def inverse_rescale(y_real_features, real_scaler):
    # Reshape the input data to 2D arrays
    real_data = y_real_features.reshape(y_real_features.shape[0], -1)

    # Inverse scale the real data using the same scaler
    real_data_scaled = real_scaler.inverse_transform(real_data)

    # Reshape the scaled data back to the original shape
    return real_data_scaled.reshape(y_real_features.shape)


_, x_time_series, x_real_features, y_real_features = data_import.load_interpolated_data()

# highly doubt dass das gut ist
# x_time_series[0, -3:, :] = x_time_series[0, -3:, :] + (x_real_features[0,:]*np.ones((x_time_series.shape[2],3))).T
# reshape x_time_series to switch the last and second last dimension
x_time_series = np.swapaxes(x_time_series, 1, 2) #[S, C, T] -> [S, T, C]
# dont use the last three features
# x_time_series = x_time_series[:, :, :-3]

x_time_series, x_real_features, y_real_features, real_scaler = rescale(x_time_series, x_real_features, y_real_features)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
num_simulations = x_time_series.shape[0]
input_size = x_time_series.shape[2]
hidden_size = 128
x_real_size = x_real_features.shape[1]
num_layers = 2
output_size = y_real_features.shape[1]
model = LSTMPredictor(input_size, hidden_size, num_layers, output_size).to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.LEARNING_RATE_DECAY)



x_time_series = torch.tensor(x_time_series).float().to(device)
x_real_features = torch.tensor(x_real_features).float().to(device)
y_real_features = torch.tensor(y_real_features).float().to(device)


# split the data into train and validation sets
test_size, val_size = 0.05, 0.1
num_samples = len(x_time_series)
test_indices = list(np.random.choice(num_samples, size=int(test_size*num_samples), replace=False))
remaining_indices = list(set(range(num_samples)) - set(test_indices))
val_indices = list(np.random.choice(remaining_indices, size=int(val_size*num_samples), replace=False))
train_indices = list(set(remaining_indices) - set(val_indices))

train_dataset = TensorDataset(x_time_series[train_indices], x_real_features[train_indices], y_real_features[train_indices])
val_dataset = TensorDataset(x_time_series[val_indices], x_real_features[val_indices], y_real_features[val_indices])

# create train and validation data loaders
batch_size = config.BATCH_SIZE
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


num_epochs = config.EPOCHS
losses, val_losses, best_val_loss = [], [], 99999
start_time = datetime.datetime.now()
for epoch in range(num_epochs):
    # train the model
    train_loss = 0
    for i, (x, q, y) in enumerate(train_loader):
        # Forward pass
        outputs = model(x, q)
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
    for i, (x, q, y) in enumerate(val_loader):
        outputs = model(x, q)
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
            torch.save(model.state_dict(), config.CHECKPOINT_PATH + "model_{}_{}.pth".format(date, time))

    # break if the validation loss hasn't improved for some epochs
    if epoch > config.EARLY_STOPPING and not best_val_loss in val_losses[-config.EARLY_STOPPING:]:
        print("Early stopping")
        break

    
# print an example predictions
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for i in range(1):
        print(x_time_series[test_indices[0:batch_size]].shape)
        print(y_real_features[test_indices[0:batch_size]].shape)
        inputs_x = x_time_series[test_indices[0:batch_size]]
        inputs_y = x_real_features[test_indices[0:batch_size]]
        # load targets to cpu
        targets = y_real_features[test_indices[0:batch_size]].cpu()
        outputs = model(inputs_x, inputs_y).cpu()
        scaled_targets = inverse_rescale(targets, real_scaler)
        scaled_outputs = inverse_rescale(outputs, real_scaler)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        for i in range(len(scaled_outputs)):
            print("pred", "real", "rel error", sep='\t')
            for j in range(len(scaled_outputs[i])):
                pred = scaled_outputs[i][j]
                real = scaled_targets[i][j]
                rel_error = (scaled_outputs[i][j] - scaled_targets[i][j])/scaled_targets[i][j]*100
                print(round(pred), round(real), str(round(rel_error, 1)) + "%", sep='\t')
            print("--------------------")


# plot the training and validation loss
plt.plot(losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Samples: {}, Data Points: {}, Batch size: {}, Learning rate: {}"
          .format(config.N_SAMPLES, x_time_series.shape[0], config.BATCH_SIZE, config.LEARNING_RATE))
plt.grid()
plt.ylim(bottom=0)
plt.legend()
plt.savefig(config.PLOT_PATH + "model_{}_{}.png".format(date, time))


# TODOS

# TODO: zip and unzip the data -> DONE
# TODO: no joint angles -> DONE
# TODO: multiple experiments
# TODO: move data to separate folder -> DONE
# TODO: data_prep divide in two files
