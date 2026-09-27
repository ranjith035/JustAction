# Local Shipping & Optimization Guide for System 1 JustAction Engine

This guide covers how to export, optimize, package, and execute your custom PyTorch **JustAction Engine** locally with zero heavy dependencies (no PyTorch overhead at runtime) using ONNX, INT8 quantization, and PyInstaller binary compilation.

---

## Architecture Overview

```
 ┌──────────────────────────┐
 │  PyTorch JevActionModel  │
 └─────────────┬────────────┘
               │ Export
               ▼
 ┌──────────────────────────┐
 │  ONNX Dynamic Engine     │
 └─────────────┬────────────┘
               │ INT8 Quantize
               ▼
 ┌──────────────────────────┐
 │  jev_model_int8.onnx     │  (< 150MB, Sub-10ms CPU/GPU Latency)
 └─────────────┬────────────┘
               │
         ┌─────┴────────────────────┐
         ▼                          ▼
 ┌───────────────┐          ┌───────────────┐
 │ Python Runner │          │ Native Binary │
 │ (onnxruntime) │          │ (PyInstaller) │
 └───────────────┘          └───────────────┘
```

---

## Step 1: Export to ONNX & Quantize (`optimize_local.py`)

Run this script to convert your trained PyTorch weights into an INT8-quantized ONNX runtime graph.

```python
import torch
import torch.onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from model import JevConfig, JevActionModel

def export_and_quantize():
    print("1. Loading PyTorch Model...")
    config = JevConfig(hidden_size=512, num_layers=8, num_heads=8)
    model = JevActionModel(config)
    model.eval()

    # Dummy inputs for graph tracing
    dummy_input_ids = torch.randint(0, 30000, (1, 128))
    dummy_mask = torch.ones((1, 128), dtype=torch.long)

    # Export to ONNX graph
    onnx_path = "jev_model.onnx"
    print(f"2. Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_mask),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["choice_probs", "noul_prob", "score_dist"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "choice_probs": {0: "batch_size"},
            "noul_prob": {0: "batch_size"},
            "score_dist": {0: "batch_size"},
        }
    )

    # INT8 Quantization (Reduces binary size by ~4x and speeds up CPU inference)
    quantized_path = "jev_model_int8.onnx"
    print(f"3. Quantizing to INT8: {quantized_path}...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QUInt8
    )
    print("Done! Optimized ONNX runtime engine created successfully.")

if __name__ == "__main__":
    export_and_quantize()
```

---

## Step 2: High-Performance Local Engine (`local_runner.py`)

Inference execution layer using `onnxruntime` with support for both CPU and CUDA acceleration.

```python
import numpy as np
import onnxruntime as ort
from typing import Dict, Any

class LocalJevEngine:
    def __init__(self, model_path: str = "jev_model_int8.onnx"):
        # Uses CUDAExecutionProvider if available, otherwise falls back to CPUExecutionProvider
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

    def predict(self, input_ids: list, attention_mask: list) -> Dict[str, Any]:
        # Convert inputs to numpy arrays
        ort_inputs = {
            "input_ids": np.array([input_ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
        }

        # Single forward pass
        choice_probs, noul_prob, score_dist = self.session.run(None, ort_inputs)

        best_choice = int(np.argmax(choice_probs[0]))
        confidence = float(choice_probs[0][best_choice])
        expected_score = float(np.sum(np.arange(len(score_dist[0])) * score_dist[0]))

        return {
            "choice_index": best_choice,
            "confidence": round(confidence, 4),
            "noul_probability": round(float(noul_prob[0]), 4),
            "expected_score": round(expected_score, 2)
        }

if __name__ == "__main__":
    engine = LocalJevEngine()
    
    # Example tokenized payload
    dummy_tokens = [101] + [500] * 126 + [102]
    dummy_mask = [1] * 128
    
    result = engine.predict(dummy_tokens, dummy_mask)
    print("Local Decision Output:", result)
```

---

## Step 3: Bundle into Standalone Native Executable

CLI entrypoint and PyInstaller compilation command to ship as a single self-contained binary.

### 1. Create CLI Script (`cli_app.py`)

```python
import json
import sys
import os
from local_runner import LocalJevEngine

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def main():
    if len(sys.argv) < 2:
        print("Usage: ./jev_engine '{\"input_ids\": [...], \"attention_mask\": [...]}'")
        sys.exit(1)

    payload = json.loads(sys.argv[1])
    model_path = get_resource_path("jev_model_int8.onnx")
    
    engine = LocalJevEngine(model_path)
    result = engine.predict(payload["input_ids"], payload["attention_mask"])
    print(json.dumps(result))

if __name__ == "__main__":
    main()
```

### 2. Build Commands

```bash
# Install packaging tools
pip install pyinstaller onnxruntime

# Compile into a single native binary
pyinstaller --onefile --add-data "jev_model_int8.onnx:." cli_app.py -n jev_engine

# Run the compiled binary locally
./dist/jev_engine '{"input_ids": [101, 500, 102], "attention_mask": [1, 1, 1]}'
```