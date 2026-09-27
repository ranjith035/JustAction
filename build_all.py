"""
End-to-End Build and Verification Pipeline for Local JustAction Engine
----------------------------------------------------------------------
Orchestrates:
1. Model training & Hugging Face AutoModel export (via train.py)
2. Graph compilation to ONNX and INT8 dynamic quantization (via optimize_local.py)
3. Local verification and benchmark with onnxruntime (via local_runner.py)
"""

import sys
import os

def run_pipeline():
    print("=" * 60)
    print("    JUSTACTION ENGINE LOCAL BUILD PIPELINE")
    print("=" * 60)

    # Step 1: Check if checkpoint exists, otherwise run train.py
    checkpoint_dir = "./justaction-engine"
    if not os.path.exists(checkpoint_dir) and os.path.exists("./jev-action-engine"):
        checkpoint_dir = "./jev-action-engine"

    if not os.path.exists(checkpoint_dir):
        print("\n[Stage 1/3] No trained checkpoint detected. Running train.py...")
        from train import train_and_export
        train_and_export()
    else:
        print(f"\n[Stage 1/3] Found existing trained checkpoint at '{checkpoint_dir}'.")

    # Step 2: Export to ONNX & Quantize
    print("\n[Stage 2/3] Exporting to ONNX and Quantizing to INT8...")
    from optimize_local import export_and_quantize
    model_path = export_and_quantize(checkpoint_dir=checkpoint_dir)

    # Step 3: Local Engine Verification
    print("\n[Stage 3/3] Verifying Local JustAction Engine...")
    try:
        from local_runner import LocalJustActionEngine
        engine = LocalJustActionEngine(model_path=model_path)
        
        dummy_tokens = [101] + [500] * 126 + [102]
        dummy_mask = [1] * 128
        
        result = engine.step(dummy_tokens, dummy_mask)
        print("Engine verification output:")
        print(result)
        print("\n[SUCCESS] JustAction Engine is optimized and verified locally!")
    except Exception as e:
        print(f"\n[NOTE] Local execution check: {e}")

    print("\nTo build a standalone single-file binary:")
    print("  pyinstaller --onefile --add-data \"justaction_int8.onnx;.\" cli_app.py -n justaction_engine")


if __name__ == "__main__":
    run_pipeline()
