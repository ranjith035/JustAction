# JustAction: High-Throughput System 1 Decision & Action Engine

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Runtime](https://img.shields.io/badge/runtime-ONNX%20%7C%20PyTorch-green.svg)](https://onnxruntime.ai/)
[![Latency](https://img.shields.io/badge/latency-%3C10ms-brightgreen.svg)]()

**JustAction** is an open-source, deterministic, non-autoregressive **System 1 Decision Engine** built for software pipelines and autonomous agent runtimes.

Unlike traditional autoregressive LLMs (System 2) that generate conversational prose token-by-token, **JustAction** executes in a single forward pass ($O(1)$ decoding steps) to directly emit strongly typed action schemas with calibrated confidence and deterministic fallback routing.

---

## Key Features

- **Non-Autoregressive Single-Pass Execution**: Eliminates the token loop. Emits decisions, rubric ratings, and boolean judgments in under 10ms.
- **Three Multi-Task Decision Primitives**:
  - **Choice**: Categorical action selection from an enumerated schema space.
  - **Score**: Rubric-based ordinal and continuous evaluations (e.g., severity, priority, risk).
  - **Noul**: Calibrated binary (True/False) judgment with associated confidence probability.
- **Calibrated Epistemic Fallback**: Built-in uncertainty thresholding (`confidence < 0.70`). Automatically activates `fallback_triggered: true` to safely route low-confidence tasks to humans or larger LLMs.
- **Zero-Prose Typed Output**: Emits pure, compact JSON adhering strictly to typed action specifications with zero conversational overhead or markdown delimiters.
- **Ultra-Lightweight Local Footprint**: Pre-quantized to **INT8 (QUInt8)** (~40 MB) for CPU/GPU deployment with zero heavy framework dependencies via `onnxruntime`.

---

## System 1 Response Schema

```json
{
  "action": "ROUTE_DATABASE",
  "confidence": 0.9421,
  "parameters": {
    "shard_id": 4,
    "priority": "HIGH"
  },
  "fallback_triggered": false
}
```

If calibrated confidence drops below the threshold (e.g., 0.70), the engine deterministically yields:
```json
{
  "action": "FALLBACK_TO_HUMAN",
  "confidence": 0.4128,
  "parameters": {},
  "fallback_triggered": true
}
```

---

## Quickstart

### 1. Installation

```bash
pip install onnxruntime
```

### 2. Python Inference (`LocalJustActionEngine`)

```python
from local_runner import LocalJustActionEngine

# Initialize the INT8 dynamic engine
engine = LocalJustActionEngine(model_path="justaction_int8.onnx")

# 1. Deterministic System 1 Action Decision
decision = engine.step(
    input_ids=[101, 2450, 3102, 102],
    parameters={"trace_id": "tx-8901"}
)
print(decision)
# => {'action': 'EXECUTE_WORKFLOW', 'confidence': 0.8921, 'parameters': {...}, 'fallback_triggered': False}

# 2. Raw Multi-Task Primitives (Choice, Noul, Score)
raw = engine.predict_raw([101, 2450, 3102, 102])
print(f"Choice Index: {raw['choice_index']}")
print(f"Noul Verdict Probability: {raw['noul_probability']}")
print(f"Expected Rubric Score: {raw['expected_score']}")
```

### 3. CLI Zero-Prose Usage

```bash
python cli_app.py '{"input_ids": [101, 500, 102], "attention_mask": [1, 1, 1]}'
```

Output:
```json
{"action":"FALLBACK_TO_HUMAN","confidence":0.2192,"parameters":{},"fallback_triggered":true}
```

---

## Build & Export Pipeline

To retrain and compile from scratch:

```bash
# Full automated pipeline (training -> ONNX export -> INT8 quantization -> verification)
python build_all.py
```

### Build a Standalone Single-File Native Binary
```bash
pip install pyinstaller
pyinstaller --onefile --add-data "justaction_int8.onnx;." cli_app.py -n justaction_engine
./dist/justaction_engine '{"input_ids": [101, 500, 102]}'
```

---

## License

This project is licensed under the MIT License.
