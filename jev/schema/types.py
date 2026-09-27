from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json

class ActionType(str, Enum):
    """Supported System 1 Action Enums."""
    NOOP = "NOOP"
    EXECUTE_WORKFLOW = "EXECUTE_WORKFLOW"
    ROUTE_DATABASE = "ROUTE_DATABASE"
    ROUTE_AUTH = "ROUTE_AUTH"
    ROUTE_PAYMENT = "ROUTE_PAYMENT"
    TRIGGER_ALERT = "TRIGGER_ALERT"
    ESCALATE_TIER2 = "ESCALATE_TIER2"
    APPLY_RATE_LIMIT = "APPLY_RATE_LIMIT"
    ARCHIVE_RECORD = "ARCHIVE_RECORD"
    FALLBACK_TO_HUMAN = "FALLBACK_TO_HUMAN"
    FALLBACK_TO_SYSTEM2 = "FALLBACK_TO_SYSTEM2"


@dataclass
class NoulDecision:
    """
    Binary judgment primitive in Jev ('Noul').
    Represents a calibrated yes/no verification decision.
    """
    verdict: bool
    probability: float
    threshold: float = 0.50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "probability": round(float(self.probability), 4)
        }


@dataclass
class ScoreDecision:
    """
    Continuous/Ordinal score primitive in Jev.
    Represents ratings on specified rubrics (e.g., severity, priority).
    """
    name: str
    value: float
    min_val: float = 0.0
    max_val: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(float(self.value), 4)
        }


@dataclass
class ActionOutput:
    """
    Deterministic System 1 Action Schema matching user execution specifications:
    {
      "action": "<ENUM_ACTION_NAME>",
      "confidence": <FLOAT_0_TO_1>,
      "parameters": { ... },
      "fallback_triggered": <BOOLEAN>
    }
    """
    action: str
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    fallback_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "confidence": round(float(self.confidence), 4),
            "parameters": self.parameters,
            "fallback_triggered": self.fallback_triggered
        }

    def to_json(self) -> str:
        """Emits zero-prose, compact valid JSON."""
        return json.dumps(self.to_dict(), separators=(',', ':'))


@dataclass
class DecisionOutput:
    """
    Complete Jev System 1 Multi-Task Output:
    Combines Choice, Scores, Nouls, and ActionOutput.
    """
    action_output: ActionOutput
    choice: str
    choice_confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    nouls: Dict[str, NoulDecision] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action_output.action,
            "confidence": round(self.action_output.confidence, 4),
            "parameters": self.action_output.parameters,
            "fallback_triggered": self.action_output.fallback_triggered,
            "choice": self.choice,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "nouls": {k: v.to_dict() for k, v in self.nouls.items()}
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
