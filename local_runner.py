import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Union

# Action Enum mappings
DEFAULT_ACTION_ENUMS = [
    "NOOP",
    "EXECUTE_WORKFLOW",
    "ROUTE_DATABASE",
    "ROUTE_AUTH",
    "ROUTE_PAYMENT",
    "TRIGGER_ALERT",
    "ESCALATE_TIER2",
    "APPLY_RATE_LIMIT",
    "ARCHIVE_RECORD",
    "UPDATE_STATE",
    "FETCH_EXTERNAL",
    "RETRY_OPERATION",
    "ROLLBACK_TRANSACTION",
    "VALIDATE_SCHEMA",
    "DISPATCH_EVENT",
    "FALLBACK_TO_HUMAN"
]

class LocalJustActionEngine:
    """
    High-Performance Local Inference Engine for JustAction System 1 Decision Model.
    Uses onnxruntime with INT8 graph execution and zero PyTorch runtime overhead.
    """
    def __init__(
        self, 
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.70,
        fallback_action: str = "FALLBACK_TO_HUMAN",
        action_names: Optional[List[str]] = None,
        verbose: bool = False
    ):
        self.confidence_threshold = confidence_threshold
        self.fallback_action = fallback_action
        self.action_names = action_names or DEFAULT_ACTION_ENUMS
        self.verbose = verbose

        # Resolve model path
        if model_path is None:
            if os.path.exists("justaction_int8.onnx"):
                model_path = "justaction_int8.onnx"
            elif os.path.exists("justaction.onnx"):
                model_path = "justaction.onnx"
            elif os.path.exists("jev_model_int8.onnx"):
                model_path = "jev_model_int8.onnx"
            else:
                model_path = "justaction_int8.onnx"

        self.model_path = model_path
        self._init_session()

    def _init_session(self):
        try:
            import onnxruntime as ort
            # Try CUDAExecutionProvider first, fall back to CPU
            available = ort.get_available_providers()
            providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in available]
            if not providers:
                providers = ['CPUExecutionProvider']

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
            if self.verbose:
                print(f"[LocalJevEngine] Loaded '{self.model_path}' using providers: {providers}")
        except ImportError:
            raise ImportError(
                "onnxruntime is required for LocalJevEngine. Install it with: pip install onnxruntime"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {self.model_path}: {e}")

    def predict_raw(
        self, 
        input_ids: List[int], 
        attention_mask: Optional[List[int]] = None,
        max_seq_len: int = 128
    ) -> Dict[str, Any]:
        """
        Runs raw single-pass inference returning Choice, Noul, and Score distributions.
        Automatically pads or truncates input to standard model sequence length.
        """
        orig_len = len(input_ids)
        if attention_mask is None:
            attention_mask = [1] * orig_len

        # Pad or truncate to fixed sequence length expected by the graph
        if orig_len < max_seq_len:
            pad_len = max_seq_len - orig_len
            padded_input_ids = list(input_ids) + [0] * pad_len
            padded_mask = list(attention_mask) + [0] * pad_len
        else:
            padded_input_ids = list(input_ids[:max_seq_len])
            padded_mask = list(attention_mask[:max_seq_len])

        ort_inputs = {
            "input_ids": np.array([padded_input_ids], dtype=np.int64),
            "attention_mask": np.array([padded_mask], dtype=np.int64),
        }

        # Single forward pass
        choice_probs, noul_prob, score_dist = self.session.run(None, ort_inputs)

        best_choice_idx = int(np.argmax(choice_probs[0]))
        confidence = float(choice_probs[0][best_choice_idx])
        
        # Expected score rating over ordinal distribution
        score_levels = len(score_dist[0])
        expected_score = float(np.sum(np.arange(score_levels) * score_dist[0]))

        return {
            "choice_index": best_choice_idx,
            "confidence": round(confidence, 4),
            "noul_probability": round(float(noul_prob[0]), 4),
            "expected_score": round(expected_score, 2),
            "choice_probs": [round(float(p), 4) for p in choice_probs[0]],
            "score_distribution": [round(float(s), 4) for s in score_dist[0]]
        }

    def step(
        self, 
        input_ids: List[int], 
        attention_mask: Optional[List[int]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Zero-Prose System 1 Decision execution matching the exact schema:
        {
          "action": "<ENUM_ACTION_NAME>",
          "confidence": <FLOAT_0_TO_1>,
          "parameters": { ... },
          "fallback_triggered": <BOOLEAN>
        }
        """
        raw = self.predict_raw(input_ids, attention_mask)
        choice_idx = raw["choice_index"]
        confidence = raw["confidence"]

        # Action mapping
        if 0 <= choice_idx < len(self.action_names):
            action = self.action_names[choice_idx]
        else:
            action = "NOOP"

        # Calibration & Fallback logic
        fallback_triggered = False
        if confidence < self.confidence_threshold:
            fallback_triggered = True
            action = self.fallback_action

        return {
            "action": action,
            "confidence": round(confidence, 4),
            "parameters": parameters or {},
            "fallback_triggered": fallback_triggered
        }

# Backward compatibility alias
LocalJevEngine = LocalJustActionEngine


if __name__ == "__main__":
    # Smoke test demo
    try:
        engine = LocalJustActionEngine()
        dummy_tokens = [101] + [500] * 126 + [102]
        dummy_mask = [1] * 128
        
        print("\nRaw Jev Multi-Task Output:")
        print(json.dumps(engine.predict_raw(dummy_tokens, dummy_mask), indent=2))
        
        print("\nDeterministic Action Engine Output (System 1 Schema):")
        print(json.dumps(engine.step(dummy_tokens, dummy_mask), separators=(',', ':')))
    except Exception as e:
        print(f"Engine initialization check: {e}")
