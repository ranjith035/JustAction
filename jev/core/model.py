import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, NamedTuple

from jev.core.config import JevConfig
from jev.core.layers import (
    SinusoidalPositionalEmbedding,
    TransformerEncoderBlock,
    AttentivePooling
)

class JevRawOutputs(NamedTuple):
    choice_logits: torch.Tensor       # Shape: (B, num_choices)
    score_predictions: torch.Tensor   # Shape: (B, num_scores)
    noul_logits: torch.Tensor         # Shape: (B, num_nouls)
    confidence_scores: torch.Tensor   # Shape: (B, 1) calibrated decision confidence
    pooled_embedding: torch.Tensor    # Shape: (B, d_model)


class JevModel(nn.Module):
    """
    Jev System 1 Non-Autoregressive Decision Engine.
    
    Processes the entire input context in a single parallel forward pass (O(1) steps)
    and directly emits typed decision primitives:
      - Choice: Categorical selection among allowed discrete actions.
      - Score: Continuous/ordinal evaluation of input state (e.g., severity, priority).
      - Noul: Calibrated binary (True/False) judgments.
      - Confidence: Calibrated scalar score [0.0, 1.0] for reliable fallback routing.
    """
    def __init__(self, config: Optional[JevConfig] = None):
        super().__init__()
        self.config = config or JevConfig()

        # Token & Positional Embeddings
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model, padding_idx=0)
        self.pos_embeddings = SinusoidalPositionalEmbedding(self.config.d_model, self.config.max_seq_len)
        self.emb_dropout = nn.Dropout(self.config.dropout)
        self.emb_norm = nn.LayerNorm(self.config.d_model)

        # Non-Autoregressive Transformer Encoder Stack
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                d_ff=self.config.d_ff,
                dropout=self.config.dropout
            )
            for _ in range(self.config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.config.d_model)

        # Attentive sequence aggregator
        self.pooler = AttentivePooling(self.config.d_model)

        # ==========================================
        # Multi-Head System 1 Output Primitives
        # ==========================================
        
        # 1. Choice Head: Discrete Categorical Actions
        self.choice_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, self.config.num_choices)
        )

        # 2. Score Head: Bounded continuous ratings (0.0 to 1.0)
        self.score_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, self.config.num_scores),
            nn.Sigmoid()  # Normalizes score to [0.0, 1.0]
        )

        # 3. Noul Head: Binary Judgments (e.g. guardrail pass/fail, policy compliance)
        self.noul_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, self.config.num_nouls)
        )

        # 4. Calibrated Confidence Estimation Head (RLCD regularized)
        # Learns to predict its own uncertainty alongside the categorical logits
        self.confidence_head = nn.Sequential(
            nn.Linear(self.config.d_model + self.config.num_choices, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, 1),
            nn.Sigmoid()
        )

        # Post-hoc Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1) * self.config.initial_temperature, requires_grad=False)

        self._init_weights()

    def _init_weights(self):
        """Standard Xavier uniform weight initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> JevRawOutputs:
        """
        Single-pass execution over batch of input tokens.
        
        Args:
            input_ids: Tensor of shape (B, S) with integer token IDs.
            attention_mask: Tensor of shape (B, S), 1 for valid tokens, 0 for pad tokens.
        """
        B, S = input_ids.shape

        # 1. Embedding + Positional Encoding
        x = self.token_embeddings(input_ids)
        x = self.pos_embeddings(x)
        x = self.emb_norm(self.emb_dropout(x))

        # Format 2D attention mask to 4D for multi-head attention: (B, 1, 1, S)
        attn_mask_4d = None
        if attention_mask is not None:
            attn_mask_4d = attention_mask.unsqueeze(1).unsqueeze(2)

        # 2. Non-autoregressive Transformer encoding
        for layer in self.layers:
            x = layer(x, mask=attn_mask_4d)
        x = self.final_norm(x)

        # 3. Attentive pooling into single state vector
        pooled = self.pooler(x, mask=attention_mask)

        # 4. Multi-head outputs
        choice_logits = self.choice_head(pooled)
        score_preds = self.score_head(pooled)
        noul_logits = self.noul_head(pooled)

        # 5. Decision Confidence Estimation
        # Combine state representation with choice distribution for calibrated certainty
        choice_probs_detached = F.softmax(choice_logits.detach() / self.temperature, dim=-1)
        conf_in = torch.cat([pooled, choice_probs_detached], dim=-1)
        predicted_conf = self.confidence_head(conf_in)

        # Temperature-scaled confidence from choice maximum probability
        max_choice_prob, _ = torch.max(choice_probs_detached, dim=-1, keepdim=True)
        
        # Ensembled calibrated confidence: blend of predictive max prob and dedicated confidence head
        calibrated_conf = 0.5 * max_choice_prob + 0.5 * predicted_conf

        return JevRawOutputs(
            choice_logits=choice_logits,
            score_predictions=score_preds,
            noul_logits=noul_logits,
            confidence_scores=calibrated_conf,
            pooled_embedding=pooled
        )

    def set_temperature(self, temp: float):
        """Set calibration temperature post-hoc."""
        self.temperature.data.fill_(max(temp, 1e-4))
