import torch
import torch.nn.functional as F
from typing import Dict, Any, Union, Optional, List
import json

from jev.core.model import JevModel
from jev.core.config import JevConfig
from jev.tokenizer.fast_tokenizer import FastTokenizer
from jev.schema.types import ActionOutput, DecisionOutput, NoulDecision, ActionType

class JevActionEngine:
    """
    Deterministic Zero-Prose Action Engine (System 1 Execution Unit).
    
    Consumes runtime state and maps it directly to typed schema actions in a single
    non-autoregressive forward pass with well-calibrated confidence and fallback routing.
    """
    def __init__(
        self, 
        model: Optional[JevModel] = None, 
        tokenizer: Optional[FastTokenizer] = None,
        config: Optional[JevConfig] = None,
        device: Optional[str] = None
    ):
        self.config = config or (model.config if model else JevConfig())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.model = model or JevModel(self.config)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = tokenizer or FastTokenizer(max_vocab_size=self.config.vocab_size)

    @torch.no_grad()
    def step(
        self, 
        runtime_state: Union[str, Dict[str, Any]], 
        parameters: Optional[Dict[str, Any]] = None
    ) -> ActionOutput:
        """
        Executes a single-pass System 1 evaluation and emits a typed ActionOutput.
        
        Args:
            runtime_state: Text string or dictionary representing current runtime observation.
            parameters: Optional execution parameters to include in the payload.
            
        Returns:
            ActionOutput conforming strictly to the requested execution schema.
        """
        if isinstance(runtime_state, dict):
            # Serialize state to canonical key=value representation
            text = " ".join(f"{k}: {v}" for k, v in runtime_state.items())
        else:
            text = str(runtime_state).strip()

        # Handle empty/unspecified runtime state immediately
        if not text:
            return ActionOutput(
                action=ActionType.NOOP.value,
                confidence=1.0,
                parameters=parameters or {},
                fallback_triggered=False
            )

        # 1. Tokenize & forward pass
        input_ids, attention_mask = self.tokenizer.batch_encode(
            [text], 
            max_length=self.config.max_seq_len, 
            device=self.device
        )
        
        raw_outputs = self.model(input_ids, attention_mask=attention_mask)

        # 2. Extract choice decision and calibrated confidence
        choice_logits = raw_outputs.choice_logits[0]  # (num_choices,)
        choice_probs = F.softmax(choice_logits / self.model.temperature, dim=-1)
        best_choice_idx = int(torch.argmax(choice_probs).item())
        
        # Calibrated confidence score
        confidence = float(raw_outputs.confidence_scores[0, 0].clamp(0.0, 1.0).item())

        # Determine choice label
        if 0 <= best_choice_idx < len(self.config.choice_labels):
            action_name = self.config.choice_labels[best_choice_idx]
        else:
            action_name = ActionType.NOOP.value

        # 3. Calibration & Fallback Logic
        fallback_triggered = False
        if confidence < self.config.confidence_threshold:
            fallback_triggered = True
            action_name = self.config.fallback_action

        return ActionOutput(
            action=action_name,
            confidence=round(confidence, 4),
            parameters=parameters or {},
            fallback_triggered=fallback_triggered
        )

    @torch.no_grad()
    def evaluate_full_decision(
        self, 
        runtime_state: Union[str, Dict[str, Any]],
        score_rubrics: Optional[List[str]] = None,
        noul_names: Optional[List[str]] = None
    ) -> DecisionOutput:
        """
        Emits all Jev System 1 primitives: Choice, Scores, Nouls, and Action payload.
        """
        action_out = self.step(runtime_state)
        
        text = json.dumps(runtime_state) if isinstance(runtime_state, dict) else str(runtime_state)
        input_ids, attention_mask = self.tokenizer.batch_encode([text], max_length=self.config.max_seq_len, device=self.device)
        raw_outputs = self.model(input_ids, attention_mask=attention_mask)

        # Scores
        rubric_keys = score_rubrics or [f"score_{i}" for i in range(self.config.num_scores)]
        scores_dict = {}
        for i, key in enumerate(rubric_keys[:self.config.num_scores]):
            scores_dict[key] = float(raw_outputs.score_predictions[0, i].item())

        # Nouls (binary verdicts)
        noul_keys = noul_names or [f"judgment_{i}" for i in range(self.config.num_nouls)]
        nouls_dict = {}
        noul_probs = torch.sigmoid(raw_outputs.noul_logits[0])
        for i, key in enumerate(noul_keys[:self.config.num_nouls]):
            prob = float(noul_probs[i].item())
            nouls_dict[key] = NoulDecision(verdict=(prob >= 0.5), probability=prob)

        return DecisionOutput(
            action_output=action_out,
            choice=action_out.action,
            choice_confidence=action_out.confidence,
            scores=scores_dict,
            nouls=nouls_dict
        )
