import json
import torch
from torch.utils.data import Dataset


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