"""
Interactive Demo for JustAction System 1 Decision Engine
--------------------------------------------------------
Test decision routing with either:
1. Token ID sequences
2. Simulated agent observation payloads
"""

import json
from local_runner import LocalJustActionEngine

def run_demo():
    print("=" * 65)
    print("           JUSTACTION SYSTEM 1 ENGINE - INTERACTIVE DEMO")
    print("=" * 65)

    # 1. Initialize engine using the quantized INT8 graph (<10ms latency)
    engine = LocalJustActionEngine(
        model_path="justaction_int8.onnx",
        confidence_threshold=0.70,
        verbose=False
    )
    print("\n[INFO] Loaded INT8 engine successfully ('justaction_int8.onnx')\n")

    # Test cases representing different runtime states
    test_cases = [
        {
            "name": "Scenario 1: Short Action Sequence (3 tokens)",
            "input_ids": [101, 500, 102],
            "attention_mask": [1, 1, 1],
            "params": {"request_id": "req-001"}
        },
        {
            "name": "Scenario 2: Database Query Payload (16 tokens)",
            "input_ids": [101, 2045, 1037, 4923, 2005, 1045, 2374, 2023, 102] + [0] * 7,
            "attention_mask": [1] * 9 + [0] * 7,
            "params": {"table": "users", "operation": "SELECT"}
        },
        {
            "name": "Scenario 3: Standard Sequence (128 tokens)",
            "input_ids": [101] + [300] * 126 + [102],
            "attention_mask": [1] * 128,
            "params": {"pipeline": "auth_middleware"}
        }
    ]

    for tc in test_cases:
        print("-" * 65)
        print(f"> {tc['name']}")
        print(f"  Input Tokens: {tc['input_ids'][:10]}... (length: {len(tc['input_ids'])})")

        # Step 1: System 1 Typed Action Decision (Deterministic JSON Schema)
        decision = engine.step(
            input_ids=tc["input_ids"],
            attention_mask=tc["attention_mask"],
            parameters=tc["params"]
        )
        print("\n  [Action Engine Output (System 1 Schema)]:")
        print(f"  {json.dumps(decision, separators=(',', ':'))}")

        # Step 2: Raw Jev/JustAction Primitives (Choice, Noul verdict, Score)
        raw = engine.predict_raw(
            input_ids=tc["input_ids"],
            attention_mask=tc["attention_mask"]
        )
        print("\n  [Multi-Task Primitives]:")
        print(f"    - Choice Index     : {raw['choice_index']} (Best Action Candidate)")
        print(f"    - Raw Confidence   : {raw['confidence']}")
        print(f"    - Noul (Boolean)   : {raw['noul_probability']} (p >= 0.5: {raw['noul_probability'] >= 0.5})")
        print(f"    - Rubric Score     : {raw['expected_score']} / 10.0")
        print(f"    - Fallback Status  : {'TRIGGERED' if decision['fallback_triggered'] else 'PASSED'}")
        print()

    print("=" * 65)
    print("Demo completed successfully!")

if __name__ == "__main__":
    run_demo()
