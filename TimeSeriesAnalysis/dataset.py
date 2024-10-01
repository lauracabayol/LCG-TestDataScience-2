import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Define a custom Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, time_step):
        """
        Args:
            data (numpy array): Time series data.
            time_step (int): Number of past observations to use for each sample.
        """
        self.data = data
        self.time_step = time_step
        self.X, self.y = self.create_sequences(self.data, self.time_step)

    def create_sequences(self, data, time_step):
        data = data.reshape(-1, 1)
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i : (i + time_step), 0])  # Use past 'time_step' points as input
            y.append(data[i + time_step, 0])  # Use the next point as the label
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Generates a single sample of data
        Args:
            idx (int): Index of the sample to retrieve.
        """
        X_sample = self.X[idx]
        y_sample = self.y[idx]
        return torch.Tensor(np.array(X_sample)), torch.Tensor(np.array(y_sample))
