import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

class TemperatureScaler(nn.Module):
    """
    Applies Post-Hoc Temperature Scaling to calibrate decision probabilities.
    Guo et al., 'On Calibration of Modern Neural Networks', ICML 2017.
    """
    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp(min=1e-3, max=100.0)
        return logits / temp

    def calibrate(
        self, 
        val_logits: torch.Tensor, 
        val_labels: torch.Tensor, 
        lr: float = 0.01, 
        max_iter: int = 50
    ) -> float:
        """
        Tunes the temperature parameter to minimize NLL on the validation dataset.
        
        Args:
            val_logits: Tensor of shape (N, num_classes)
            val_labels: Tensor of shape (N,)
            
        Returns:
            Optimal temperature value
        """
        device = val_logits.device
        self.to(device)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return float(self.temperature.item())
