from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class JevConfig:
    """
    Configuration parameters for Jev System 1 Architecture.
    """
    # Architecture hyperparameters
    vocab_size: int = 4096
    max_seq_len: int = 256
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"

    # Multi-head System 1 Decision Outputs
    num_choices: int = 10                  # Dimension of primary Choice/Action categorical space
    num_scores: int = 2                    # Number of bounded scalar outputs (e.g. urgency, severity)
    num_nouls: int = 2                     # Number of binary (yes/no) calibrated judgments
    choice_labels: List[str] = field(default_factory=lambda: [
        "NOOP",
        "EXECUTE_WORKFLOW",
        "ROUTE_DATABASE",
        "ROUTE_AUTH",
        "ROUTE_PAYMENT",
        "TRIGGER_ALERT",
        "ESCALATE_TIER2",
        "APPLY_RATE_LIMIT",
        "ARCHIVE_RECORD",
        "FALLBACK_TO_HUMAN"
    ])

    # Calibration & Decision Fallback
    confidence_threshold: float = 0.70     # Below this, fallback_triggered is activated
    fallback_action: str = "FALLBACK_TO_HUMAN"
    initial_temperature: float = 1.0       # Temperature scaling parameter for calibrated probabilities
    use_calibrated_confidence_head: bool = True
