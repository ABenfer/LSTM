import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import datetime
import config
import importlib
import os
from LSTMPredictor import LSTMPredictor
from JsonDataset import JsonDataset


importlib.reload(config)


def print_config_variables():
    importlib.reload(config)
    print("ver lstm = 0.5.9")
    for key, value in config.__dict__.items():
        if not key.startswith("__"):
            print(key, value)


def get_current_date_and_time():
    now = datetime.datetime.now()
    date = now.strftime("%Y_%m_%d")
    time = now.strftime("%H_%M_%S")
    return date, time


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


def split_data_into_sets(samples_ts, val_size=0.1):
    S = np.array(samples_ts).shape[0]
    val_indices = list(np.random.choice(S, size=int(val_size*S), replace=False))
    train_indices = list(set(range(S)) - set(val_indices))
    return train_indices, val_indices


def create_data_loaders(dataset, indices, batch_size):
    return DataLoader([dataset[i] for i in indices], batch_size=batch_size, shuffle=True)


def define_model(C, Q, O, E, device, hidden_size=128, num_layers=2):
    model = LSTMPredictor(input_size=C, hidden_size=128, num_layers=2, output_size=O, q_size=Q, n_experiments=E).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.LEARNING_RATE_DECAY)

    return model, criterion, optimizer, scheduler


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xs, qs, y in loader:
        xs, qs, y = xs.to(device), qs.to(device), y.to(device)
        outputs = model(xs, qs)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xs, qs, y in loader:
            xs, qs, y = xs.to(device), qs.to(device), y.to(device)
            outputs = model(xs, qs)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(loader)


def print_progress(epoch, num_epochs, train_loss, val_loss, start_time, best_val_loss):
    estimated_time_finished = start_time + (datetime.datetime.now() - start_time) * (num_epochs / (epoch + 1))
    is_best = val_loss < best_val_loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {datetime.datetime.now():%H:%M:%S}, Estimated Finish: {estimated_time_finished:%H:%M:%S} {'*' if is_best else ''}")
    return is_best


def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, date, time):
    losses, val_losses = [], []
    best_val_loss = float('inf')
    start_time = datetime.datetime.now()

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        losses.append(train_loss)

        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        is_best = print_progress(epoch, num_epochs, train_loss, val_loss, start_time, best_val_loss)
        if is_best:
            best_val_loss = val_loss
            if epoch > 20:
                torch.save(model.state_dict(), config.CHECKPOINT_PATH + "model_{}_{}.pt".format(date, time))

        if epoch > config.EARLY_STOPPING and best_val_loss not in val_losses[-config.EARLY_STOPPING:]:
            print("Early stopping")
            break

        # if loss is twice as large as last one, load last best model
        if epoch > 30 and val_losses[-1] > 2 * val_losses[-2]:
            print("Loss is too large, loading last best model")
            model.load_state_dict(torch.load(config.CHECKPOINT_PATH + "model_{}_{}.pt".format(date, time)))
    
    return losses, val_losses



def print_example_predictions(model, loader, norm_parameter, device):
    with torch.no_grad():
        xs, qs, y = next(iter(loader))
        xs = xs.to(device)
        qs = qs.to(device)
        y = y.to(device)
        norm_parameter = {key: value.to(device) for key, value in norm_parameter.items()}
        outputs = model(xs, qs)
        outputs = inverse_normilize(outputs, norm_parameter)
        y = inverse_normilize(y, norm_parameter)
        for estimation, real in zip(outputs, y):
            print("Est", "Real", "Error (%)", sep="\t")
            for el_e, el_r in zip(estimation, real):
                el_e = el_e.item()
                el_r = el_r.item()
                rel_error = round(100 * (el_e - el_r) / el_r, 1)
                print(round(el_e, 1), round(el_r, 1), rel_error, sep="\t")
            print("------")
   

def plot_losses(losses, val_losses, date, time, S, T):
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


def main():
    print_config_variables()
    date, time = get_current_date_and_time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    json_files = np.sort([config.SIMS_PATH + el for el in os.listdir(config.SIMS_PATH) if el.endswith('.json')])[:]
    samples_ts, samples_q, y, norm_parameter = load_and_normalize_data(json_files)

    samples_ts = np.swapaxes(samples_ts, 2, 3) #[S, E, C, T] -> [S, E, T, C]

    S, E, T, C, Q, O = samples_ts.shape[0], samples_ts.shape[1], samples_ts.shape[2], samples_ts.shape[3], samples_q.shape[2], y.shape[1]
    print("S: {}, E: {}, T: {}, C: {}, Q: {}, O: {}".format(S, E, T, C, Q, O))

    # create train and validation data loaders
    train_indices, val_indices = split_data_into_sets(samples_ts)
    train_dataset = [(samples_ts[i], samples_q[i], y[i]) for i in range(S)]
    train_loader = create_data_loaders(train_dataset, train_indices, config.BATCH_SIZE)
    val_loader = create_data_loaders(train_dataset, val_indices, config.BATCH_SIZE)

    # Define the model
    model, criterion, optimizer, scheduler = define_model(C, Q, O, E, device)
    losses, val_losses = train_model(model, train_loader, val_loader, config.EPOCHS, criterion, optimizer, device, date, time)

    print_example_predictions(model, val_loader, norm_parameter, device)
    plot_losses(losses, val_losses, date, time, S, T)


if __name__ == "__main__":
    main()
    




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