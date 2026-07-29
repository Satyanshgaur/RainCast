import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss Function for Rain Rate Narrowcasting.
    Combines Data Loss (MSE/Huber) with ITU-R P.618 Propagation Physics Penalty.
    
    Physics Equation:
      A_predicted = k * (R_hat ** alpha) * L_eff
    Penalty:
      Loss_physics = Mean((A_predicted - A_excess)**2)
      
    Total Loss = Loss_data + lambda_physics * Loss_physics
    """
    def __init__(self, lambda_physics: float = 0.05):
        super(PhysicsInformedLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()

    def forward(self, r_pred, r_true, itu_k, itu_alpha, l_eff, a_excess):
        # Data MSE loss
        loss_data = self.mse(r_pred, r_true)
        
        # Physics Forward Prediction
        # Clamp r_pred to non-negative to avoid complex power outputs
        r_clamp = torch.clamp(r_pred, min=0.0)
        
        # A_phys = k * (r_clamp ** alpha) * l_eff
        # itu_k, itu_alpha, l_eff are (Batch, 1) tensors
        a_phys = itu_k * (r_clamp ** itu_alpha) * l_eff
        
        # Physics Constraint Penalty: Predicted Attenuation must equal observed excess attenuation
        loss_physics = self.mse(a_phys, a_excess)
        
        total_loss = loss_data + self.lambda_physics * loss_physics
        return total_loss, loss_data, loss_physics


class QuantileLoss(nn.Module):
    """
    Pinball / Quantile Loss for Probabilistic Interval Prediction.
    Evaluates percentiles q in [0.10, 0.50, 0.90].
    """
    def __init__(self, quantiles=[0.10, 0.50, 0.90]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        # preds shape: (Batch, 3) representing [q0.10, q0.50, q0.90]
        # target shape: (Batch, 1)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i:i+1]
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(torch.mean(loss))
        return torch.sum(torch.stack(losses))


class HeteroscedasticNLLLoss(nn.Module):
    """
    Heteroscedastic Gaussian Negative Log-Likelihood Loss.
    Predicts both Mean (mu) and Log-Variance (log_var) to model aleatoric uncertainty.
    """
    def __init__(self):
        super(HeteroscedasticNLLLoss, self).__init__()

    def forward(self, mean, log_var, target):
        precision = torch.exp(-log_var)
        loss = 0.5 * (precision * (target - mean) ** 2 + log_var)
        return torch.mean(loss)


class ChippedCausalPadding1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(ChippedCausalPadding1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class ProbabilisticTemporalBlock(nn.Module):
    """
    Dilated Causal Residual Block with Monte Carlo Spatial Dropout.
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, dropout: float = 0.2):
        super(ProbabilisticTemporalBlock, self).__init__()
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

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class PhysicsInformedProbabilisticTCN(nn.Module):
    """
    Physics-Informed Probabilistic 1D TCN Architecture.
    Outputs:
      1. Rain Event Probability (Classification)
      2. Point Rain Rate Estimate / Mean (mu)
      3. Log-Variance (log_var) for Heteroscedastic Uncertainty
      4. Quantiles [q0.10, q0.50, q0.90] for Prediction Intervals (+/- bounds)
    """
    def __init__(self, in_channels: int = 30, num_channels: list = None, kernel_size: int = 3, dropout: float = 0.2):
        super(PhysicsInformedProbabilisticTCN, self).__init__()
        if num_channels is None:
            num_channels = [32, 32, 64, 64, 128]

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                ProbabilisticTemporalBlock(
                    in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, dropout=dropout
                )
            )

        self.tcn = nn.Sequential(*layers)
        last_ch = num_channels[-1]
        
        self.fc_shared = nn.Sequential(
            nn.Linear(last_ch, 64),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        # Classification Head
        self.clf_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Regression Point Estimate / Mean Head
        self.reg_mean_head = nn.Linear(64, 1)
        
        # Log Variance Head (Aleatoric Uncertainty)
        self.reg_logvar_head = nn.Linear(64, 1)
        
        # Quantile Head (q0.10, q0.50, q0.90)
        self.quantile_head = nn.Linear(64, 3)

    def forward(self, x):
        output = self.tcn(x)
        last_step_feat = output[:, :, -1]
        emb = self.fc_shared(last_step_feat)
        
        prob = self.clf_head(emb)
        mu = self.reg_mean_head(emb)
        log_var = self.reg_logvar_head(emb)
        quantiles = self.quantile_head(emb)
        
        return prob, mu, log_var, quantiles

    def predict_mc_dropout(self, test_loader, device, num_samples: int = 30):
        """
        Monte Carlo Dropout Inference for Epistemic Uncertainty Estimation (Bayesian NN).
        Enables dropout during evaluation and computes mean and standard deviation across passes.
        """
        self.train()  # Keep dropout active for MC sampling
        all_pass_means = []  # Shape: (num_samples, total_samples)
        
        with torch.no_grad():
            for _ in range(num_samples):
                sample_preds = []
                for X_b, _, _ in test_loader:
                    X_b = X_b.to(device)
                    _, mu, _, _ = self.forward(X_b)
                    sample_preds.extend(mu.cpu().numpy().flatten())
                all_pass_means.append(sample_preds)
                
        all_pass_means = np.array(all_pass_means)  # Shape: (num_samples, N)
        mc_mean = np.mean(all_pass_means, axis=0)
        mc_std = np.std(all_pass_means, axis=0)
        return mc_mean, mc_std
