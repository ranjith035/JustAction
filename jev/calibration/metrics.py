import torch
import numpy as np
from typing import Dict, Tuple

def compute_ece(
    probs: torch.Tensor, 
    labels: torch.Tensor, 
    n_bins: int = 10
) -> Tuple[float, float, Dict]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        probs: Predicted probability distributions (N, num_classes) or confidence scores (N,).
        labels: Ground truth class indices (N,).
        n_bins: Number of confidence bins (standard is 10 or 15).
        
    Returns:
        (ece, mce, reliability_data)
    """
    if probs.dim() == 2:
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)
    else:
        confidences = probs
        accuracies = labels.bool()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    
    bin_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper) if i > 0 else confidences.ge(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            gap = abs(avg_confidence_in_bin - accuracy_in_bin)
            
            ece += gap * prop_in_bin
            mce = max(mce, gap)
            
            bin_data.append({
                "bin_idx": i,
                "range": (bin_lower.item(), bin_upper.item()),
                "sample_ratio": prop_in_bin,
                "accuracy": accuracy_in_bin,
                "avg_confidence": avg_confidence_in_bin,
                "calibration_gap": gap
            })

    return float(ece), float(mce), {"bins": bin_data}


def compute_brier_score(probs: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """
    Computes multi-class Brier score (mean squared error between predicted probabilities
    and one-hot encoded ground truth).
    Optimal score is 0.0 (perfectly calibrated and accurate).
    """
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    
    brier = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item()
    return float(brier)
