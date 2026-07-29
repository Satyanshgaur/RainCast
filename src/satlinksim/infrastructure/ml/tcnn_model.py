import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TemporalSequenceDataset(Dataset):
    """
    Constructs 60-minute sliding sequence windows from raw telemetry time series.
    Input shape per window: (channels, seq_len=60)
    Target per window: (rain_class_label, rain_rate_mmh) at timestep t
    """
    def __init__(self, feature_matrix: np.ndarray, rain_rate_targets: np.ndarray, seq_len: int = 60):
        self.seq_len = seq_len
        self.X_windows = []
        self.y_class = []
        self.y_reg = []

        n_steps, n_features = feature_matrix.shape
        if n_steps > seq_len:
            for t in range(seq_len - 1, n_steps):
                window = feature_matrix[t - seq_len + 1 : t + 1, :]  # Shape: (60, C)
                # Transpose to (C, 60) for 1D Conv input
                self.X_windows.append(window.T)
                
                target_rate = rain_rate_targets[t]
                self.y_reg.append(target_rate)
                self.y_class.append(1.0 if target_rate > 0.1 else 0.0)

        self.X_windows = torch.tensor(np.array(self.X_windows), dtype=torch.float32)
        self.y_class = torch.tensor(np.array(self.y_class), dtype=torch.float32).unsqueeze(1)
        self.y_reg = torch.tensor(np.array(self.y_reg), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_windows)

    def __getitem__(self, idx):
        return self.X_windows[idx], self.y_class[idx], self.y_reg[idx]


class TemporalCNN(nn.Module):
    """
    1D Temporal Convolutional Network for 60-minute telemetry sequence learning.
    Architecture:
      Conv1D(in_channels, 32, k=5, p=2) -> BatchNorm -> ReLU
      Conv1D(32, 64, k=5, s=2, p=2)      -> BatchNorm -> ReLU  (Seq length: 60 -> 30)
      Conv1D(64, 128, k=3, s=2, p=1)     -> BatchNorm -> ReLU  (Seq length: 30 -> 15)
      Flatten (128 * 15 = 1920)
      FC(1920 -> 128) -> Dropout -> ReLU
      Head 1 (Classification): Linear(128 -> 1) -> Sigmoid
      Head 2 (Regression):     Linear(128 -> 1)
    """
    def __init__(self, in_channels: int = 8, seq_len: int = 60):
        super(TemporalCNN, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Reduced sequence length calculation: 60 -> 30 -> 15
        flattened_dim = 128 * 15
        
        self.fc_shared = nn.Sequential(
            nn.Linear(flattened_dim, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        
        self.clf_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.reg_head = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (B, C, 60)
        feat = self.conv_block(x)
        feat = feat.view(feat.size(0), -1)  # Flatten
        shared_emb = self.fc_shared(feat)
        
        prob = self.clf_head(shared_emb)
        rate = self.reg_head(shared_emb)
        return prob, rate
