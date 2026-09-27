import torch
import torch.nn as nn
import torch.nn.functional as F

class RLCDLoss(nn.Module):
    """
    Reinforcement Learning for Calibrated Decisions (RLCD) Loss.
    
    A composite loss objective designed to train System 1 decision models:
    1. Task Accuracy: Cross-Entropy for discrete choice selection.
    2. Score Regression: Smooth L1 / MSE loss for continuous scores.
    3. Binary Judgment: BCE for binary Noul decisions.
    4. Calibration Penalty: Heavily penalizes overconfident errors and
       underconfident correct actions via Brier alignment and confidence-margin loss.
    """
    def __init__(
        self,
        choice_weight: float = 1.0,
        score_weight: float = 0.5,
        noul_weight: float = 0.5,
        brier_weight: float = 0.4,
        confidence_weight: float = 0.6,
        overconfidence_penalty: float = 2.0
    ):
        super().__init__()
        self.choice_weight = choice_weight
        self.score_weight = score_weight
        self.noul_weight = noul_weight
        self.brier_weight = brier_weight
        self.confidence_weight = confidence_weight
        self.overconfidence_penalty = overconfidence_penalty

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        choice_logits: torch.Tensor,
        target_choices: torch.Tensor,
        score_preds: torch.Tensor,
        target_scores: torch.Tensor,
        noul_logits: torch.Tensor,
        target_nouls: torch.Tensor,
        confidence_preds: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the complete multi-task calibrated loss.
        """
        # 1. Primary task loss
        loss_choice = self.ce_loss(choice_logits, target_choices)
        loss_score = self.mse_loss(score_preds, target_scores)
        loss_noul = self.bce_loss(noul_logits, target_nouls)

        # 2. Brier score for probability distribution calibration
        num_classes = choice_logits.size(-1)
        choice_probs = F.softmax(choice_logits, dim=-1)
        one_hot_targets = F.one_hot(target_choices, num_classes=num_classes).float()
        loss_brier = torch.mean(torch.sum((choice_probs - one_hot_targets) ** 2, dim=-1))

        # 3. Decision correctness indicator (1 if argmax matches target, 0 otherwise)
        predicted_classes = torch.argmax(choice_logits, dim=-1)
        is_correct = (predicted_classes == target_choices).float().unsqueeze(-1)  # (B, 1)

        # 4. Asymmetric Calibration Loss:
        # Penalizes overconfidence when the decision is wrong
        # Error when wrong: confidence^2 * penalty
        conf_error = torch.abs(confidence_preds - is_correct)
        asymmetric_weight = torch.where(
            is_correct == 0.0,
            torch.tensor(self.overconfidence_penalty, device=confidence_preds.device),
            torch.tensor(1.0, device=confidence_preds.device)
        )
        loss_confidence = torch.mean(asymmetric_weight * (conf_error ** 2))

        # Total combined RLCD objective
        total_loss = (
            self.choice_weight * loss_choice +
            self.score_weight * loss_score +
            self.noul_weight * loss_noul +
            self.brier_weight * loss_brier +
            self.confidence_weight * loss_confidence
        )

        return total_loss
