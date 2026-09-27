import os
import sys
import json
from local_runner import LocalJustActionEngine

def get_resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, compatible with both local python and PyInstaller --onefile
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def main():
    # 1. Read input payload from sys.argv or stdin
    payload_str = ""
    if len(sys.argv) > 1:
        payload_str = sys.argv[1].strip()
    elif not sys.stdin.isatty():
        payload_str = sys.stdin.read().strip()

    if not payload_str:
        # Default zero-prose fallback when no input provided
        output = {
            "action": "NOOP",
            "confidence": 1.0,
            "parameters": {},
            "fallback_triggered": False
        }
        print(json.dumps(output, separators=(',', ':')))
        return

    try:
        data = json.loads(payload_str)
    except json.JSONDecodeError:
        output = {
            "action": "FALLBACK_TO_HUMAN",
            "confidence": 0.0,
            "parameters": {"error": "Invalid JSON input"},
            "fallback_triggered": True
        }
        print(json.dumps(output, separators=(',', ':')))
        return

    # 2. Extract tokens or state
    input_ids = data.get("input_ids", [])
    attention_mask = data.get("attention_mask", None)
    parameters = data.get("parameters", {})

    # If input is empty, emit NOOP
    if not input_ids:
        output = {
            "action": "NOOP",
            "confidence": 1.0,
            "parameters": parameters,
            "fallback_triggered": False
        }
        print(json.dumps(output, separators=(',', ':')))
        return

    # 3. Resolve model path
    model_path = get_resource_path("justaction_int8.onnx")
    if not os.path.exists(model_path):
        model_path = get_resource_path("justaction.onnx")
    if not os.path.exists(model_path):
        model_path = get_resource_path("jev_model_int8.onnx")

    try:
        engine = LocalJustActionEngine(model_path=model_path)
        decision = engine.step(input_ids=input_ids, attention_mask=attention_mask, parameters=parameters)
        # Emit zero-prose valid JSON
        print(json.dumps(decision, separators=(',', ':')))
    except Exception as e:
        output = {
            "action": "FALLBACK_TO_HUMAN",
            "confidence": 0.0,
            "parameters": {"error": str(e)},
            "fallback_triggered": True
        }
        print(json.dumps(output, separators=(',', ':')))


if __name__ == "__main__":
    main()
