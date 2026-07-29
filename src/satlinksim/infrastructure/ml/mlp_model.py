import torch
import torch.nn as nn

class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron (Feedforward Deep Neural Network) for Rain Rate Narrowcasting.
    Architecture:
      Input Layer -> Linear(in_dim, 256) -> BatchNorm -> ReLU -> Dropout(0.2)
                  -> Linear(256, 128)   -> BatchNorm -> ReLU -> Dropout(0.2)
                  -> Linear(128, 64)    -> BatchNorm -> ReLU
                  -> Classification Head: Linear(64, 1) -> Sigmoid
                  -> Regression Head:     Linear(64, 1)
    """
    def __init__(self, in_features: int, dropout: float = 0.2):
        super(MLPModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.clf_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.reg_head = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (Batch, in_features)
        feat = self.encoder(x)
        prob = self.clf_head(feat)
        rate = self.reg_head(feat)
        return prob, rate
