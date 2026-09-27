import json
from typing import Dict, Any, Type, Union
from jev.schema.types import ActionOutput, DecisionOutput

class ZeroProseSerializer:
    """
    Enforces zero-prose, strictly typed JSON serialization.
    Ensures no markdown wrappers, no commentary, and valid schema keys only.
    """
    @staticmethod
    def serialize_action(
        action: str, 
        confidence: float, 
        parameters: Dict[str, Any], 
        fallback_triggered: bool
    ) -> str:
        payload = {
            "action": action,
            "confidence": round(float(confidence), 4),
            "parameters": parameters,
            "fallback_triggered": fallback_triggered
        }
        return json.dumps(payload, separators=(',', ':'))

    @staticmethod
    def validate_action_payload(raw_json: str) -> ActionOutput:
        data = json.loads(raw_json)
        required_keys = {"action", "confidence", "parameters", "fallback_triggered"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"Schema violation: missing fields {missing}")
        return ActionOutput(
            action=str(data["action"]),
            confidence=float(data["confidence"]),
            parameters=dict(data["parameters"]),
            fallback_triggered=bool(data["fallback_triggered"])
        )
