import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torch.utils.data import Dataset
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


class ChippedCausalPadding1d(nn.Module):
    """
    Causal padding layer ensuring predictions at time t depend strictly on timesteps <= t.
    Trims the right-side padding added by Conv1d.
    """
    def __init__(self, chomp_size: int):
        super(ChippedCausalPadding1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Bai et al. (2018) Dilated Causal Residual Block.
    Consists of two dilated causal 1D convolutions, batch normalization, ReLU, dropout,
    and a residual connection (1x1 conv if channel dimensions differ).
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = ChippedCausalPadding1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = ChippedCausalPadding1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )

        # 1x1 Conv for residual connection matching
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalCNN(nn.Module):
    """
    Bai et al. (2018) Temporal Convolutional Network (TCN) Architecture.
    Constructed from stacked dilated causal residual blocks with exponential dilation factors [1, 2, 4, 8, 16].
    
    Receptive Field = 1 + 2 * (kernel_size - 1) * sum(dilations)
    With kernel_size=3 and dilations=[1, 2, 4, 8, 16], Receptive Field = 125 timesteps (> 60 minutes).
    """
    def __init__(self, in_channels: int = 8, num_channels: list = None, kernel_size: int = 3, dropout: float = 0.2):
        super(TemporalCNN, self).__init__()
        if num_channels is None:
            num_channels = [32, 32, 64, 64, 128]

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, dropout=dropout
                )
            )

        self.tcn = nn.Sequential(*layers)
        
        # Dual prediction heads operating on the final causal timestep t=60 representation
        last_ch = num_channels[-1]
        
        self.fc_shared = nn.Sequential(
            nn.Linear(last_ch, 64),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        self.clf_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.reg_head = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (Batch, Channels, SeqLen=60)
        output = self.tcn(x)  # Shape: (Batch, 128, 60)
        
        # Extract representation at the final causal timestep t (index -1)
        last_step_feat = output[:, :, -1]  # Shape: (Batch, 128)
        
        emb = self.fc_shared(last_step_feat)
        
        prob = self.clf_head(emb)
        rate = self.reg_head(emb)
        return prob, rate
