import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

class JustActionConfig(PretrainedConfig):
    model_type = "justaction_engine"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_choices=32,
        max_score_levels=10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_choices = max_choices
        self.max_score_levels = max_score_levels


class JustActionModel(PreTrainedModel):
    config_class = JustActionConfig

    def __init__(self, config: JustActionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # 1. Choice Head (Multi-class Action Classification)
        self.choice_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.max_choices)
        )

        # 2. Noul Head (Calibrated Boolean Probability)
        self.noul_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # 3. Score Head (Ordinal Rating Scale)
        self.score_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.max_score_levels)
        )

        self.post_init()

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        target_choice=None, 
        target_noul=None, 
        target_score=None,
        return_dict=None
    ):
        if return_dict is None:
            return_dict = self.config.use_return_dict if hasattr(self.config, "use_return_dict") else True

        x = self.embeddings(input_ids)
        encoder_output = self.encoder(x)

        # Mean pooling context over valid token mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_output.size()).float()
            pooled_state = torch.sum(encoder_output * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        else:
            pooled_state = encoder_output.mean(dim=1)

        choice_logits = self.choice_head(pooled_state)
        noul_prob = self.noul_head(pooled_state).view(-1)
        score_logits = self.score_head(pooled_state)

        choice_probs = F.softmax(choice_logits, dim=-1)
        score_dist = F.softmax(score_logits, dim=-1)

        total_loss = None
        if target_choice is not None and target_noul is not None and target_score is not None:
            # Multi-Task Loss + Brier Score Calibration Penalty
            loss_choice = F.cross_entropy(choice_logits, target_choice)
            loss_noul = F.mse_loss(noul_prob, target_noul.float().view(-1))
            loss_score = F.cross_entropy(score_logits, target_score)
            total_loss = loss_choice + loss_noul + loss_score

        if not return_dict:
            return choice_probs, noul_prob, score_dist

        return {
            "loss": total_loss,
            "choice_probs": choice_probs,
            "noul_prob": noul_prob,
            "score_dist": score_dist
        }

# Aliases for backward compatibility
JevConfig = JustActionConfig
JevActionModel = JustActionModel