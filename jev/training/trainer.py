import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
import os

from jev.core.model import JevModel
from jev.core.config import JevConfig
from jev.tokenizer.fast_tokenizer import FastTokenizer
from jev.calibration.rlcd_loss import RLCDLoss
from jev.calibration.metrics import compute_ece, compute_brier_score
from jev.calibration.temperature import TemperatureScaler

class JevDecisionDataset(Dataset):
    """
    Dataset container for Jev multi-task training.
    """
    def __init__(
        self,
        texts: List[str],
        choices: List[int],
        scores: Optional[List[List[float]]] = None,
        nouls: Optional[List[List[float]]] = None
    ):
        self.texts = texts
        self.choices = torch.tensor(choices, dtype=torch.long)
        
        num_samples = len(texts)
        if scores is not None:
            self.scores = torch.tensor(scores, dtype=torch.float)
        else:
            self.scores = torch.zeros((num_samples, 2), dtype=torch.float)
            
        if nouls is not None:
            self.nouls = torch.tensor(nouls, dtype=torch.float)
        else:
            self.nouls = torch.zeros((num_samples, 2), dtype=torch.float)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.texts[idx], self.choices[idx], self.scores[idx], self.nouls[idx]


class JevTrainer:
    """
    Trainer for Jev System 1 models utilizing RLCD (Reinforcement Learning for Calibrated Decisions).
    """
    def __init__(
        self,
        model: JevModel,
        tokenizer: FastTokenizer,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        device: Optional[str] = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        
        self.criterion = RLCDLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )

    def collate_fn(self, batch):
        texts, choices, scores, nouls = zip(*batch)
        input_ids, attention_mask = self.tokenizer.batch_encode(
            list(texts), 
            max_length=self.model.config.max_seq_len,
            device=self.device
        )
        return (
            input_ids,
            attention_mask,
            torch.stack(choices).to(self.device),
            torch.stack(scores).to(self.device),
            torch.stack(nouls).to(self.device)
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for input_ids, mask, choices, scores, nouls in dataloader:
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask=mask)
            
            loss = self.criterion(
                choice_logits=outputs.choice_logits,
                target_choices=choices,
                score_preds=outputs.score_predictions,
                target_scores=scores,
                noul_logits=outputs.noul_logits,
                target_nouls=nouls,
                confidence_preds=outputs.confidence_scores
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_logits = []
        all_choices = []
        all_confidences = []

        for input_ids, mask, choices, _, _ in dataloader:
            outputs = self.model(input_ids, attention_mask=mask)
            all_logits.append(outputs.choice_logits.cpu())
            all_choices.append(choices.cpu())
            all_confidences.append(outputs.confidence_scores.cpu())

        cat_logits = torch.cat(all_logits, dim=0)
        cat_choices = torch.cat(all_choices, dim=0)
        
        probs = torch.softmax(cat_logits / self.model.temperature.cpu(), dim=-1)
        preds = torch.argmax(probs, dim=-1)
        accuracy = (preds == cat_choices).float().mean().item()

        ece, mce, _ = compute_ece(probs, cat_choices)
        brier = compute_brier_score(probs, cat_choices, num_classes=self.model.config.num_choices)

        return {
            "accuracy": float(accuracy),
            "ece": float(ece),
            "mce": float(mce),
            "brier_score": float(brier)
        }

    def calibrate_temperature(self, val_dataloader: DataLoader) -> float:
        """
        Tunes the temperature parameter using post-hoc scaling on the validation set.
        """
        self.model.eval()
        val_logits = []
        val_choices = []

        with torch.no_grad():
            for input_ids, mask, choices, _, _ in val_dataloader:
                outputs = self.model(input_ids, attention_mask=mask)
                val_logits.append(outputs.choice_logits)
                val_choices.append(choices)

        all_logits = torch.cat(val_logits, dim=0)
        all_choices = torch.cat(val_choices, dim=0)

        scaler = TemperatureScaler(init_temperature=1.0)
        optimal_temp = scaler.calibrate(all_logits, all_choices)
        self.model.set_temperature(optimal_temp)
        return optimal_temp

    def save_checkpoint(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(directory, "model.pt"))
        self.tokenizer.save_vocab(os.path.join(directory, "vocab.json"))
