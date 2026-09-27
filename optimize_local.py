import os
import torch
import torch.nn as nn
from model import JustActionConfig, JustActionModel

class ExportWrapper(nn.Module):
    """
    Wrapper around JustActionModel ensuring standard tuple outputs for clean ONNX tracing.
    """
    def __init__(self, model: JustActionModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Explicit forward returning tuple of tensors: (choice_probs, noul_prob, score_dist)
        return self.model(input_ids, attention_mask=attention_mask, return_dict=False)


def export_and_quantize(
    checkpoint_dir: str = "./justaction-engine",
    onnx_output_path: str = "justaction.onnx",
    quantized_output_path: str = "justaction_int8.onnx"
):
    print("=" * 60)
    print("Step 1: Preparing PyTorch JustActionModel...")
    print("=" * 60)

    # 1. Load trained checkpoint if exists, otherwise instantiate architecture
    if not os.path.exists(checkpoint_dir) and os.path.exists("./jev-action-engine"):
        checkpoint_dir = "./jev-action-engine"

    if os.path.isdir(checkpoint_dir) and (os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")) or os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin"))):
        print(f"Loading trained weights from '{checkpoint_dir}'...")
        model = JustActionModel.from_pretrained(checkpoint_dir)
    else:
        print("No existing checkpoint found. Initializing JustActionModel with default config...")
        config = JustActionConfig(
            vocab_size=32000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            max_choices=16,
            max_score_levels=10
        )
        model = JustActionModel(config)

    model.eval()
    export_model = ExportWrapper(model)
    export_model.eval()

    # 2. Prepare dummy inputs for tracing
    batch_size = 1
    seq_len = 128
    dummy_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    dummy_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    # 2. Always export an optimized TorchScript graph
    ts_path = "justaction_traced.pt"
    print(f"\nStep 2: Compiling TorchScript optimized graph -> '{ts_path}'...")
    try:
        traced_model = torch.jit.trace(export_model, (dummy_input_ids, dummy_mask))
        traced_model = torch.jit.freeze(traced_model)
        traced_model.save(ts_path)
        ts_size_mb = os.path.getsize(ts_path) / (1024 * 1024)
        print(f"TorchScript graph compiled successfully! Size: {ts_size_mb:.2f} MB")
    except Exception as e:
        print(f"[WARNING] TorchScript compilation failed: {e}")

    # 3. ONNX Export and Quantization
    print(f"\nStep 3: Exporting PyTorch model to ONNX -> '{onnx_output_path}'...")
    has_onnx = False
    try:
        import onnx
        torch.onnx.export(
            export_model,
            (dummy_input_ids, dummy_mask),
            onnx_output_path,
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
        raw_size_mb = os.path.getsize(onnx_output_path) / (1024 * 1024)
        print(f"ONNX graph exported successfully! Size: {raw_size_mb:.2f} MB")
        has_onnx = True
    except (ImportError, Exception) as e:
        print(f"[INFO] ONNX export unavailable ({e}). Continuing with TorchScript runtime.")

    if has_onnx:
        print(f"\nStep 4: Quantizing graph to INT8 (QUInt8) -> '{quantized_output_path}'...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                model_input=onnx_output_path,
                model_output=quantized_output_path,
                weight_type=QuantType.QUInt8
            )
            quant_size_mb = os.path.getsize(quantized_output_path) / (1024 * 1024)
            print(f"Quantization complete! Size: {quant_size_mb:.2f} MB ({raw_size_mb / max(quant_size_mb, 1e-3):.1f}x compression)")
            return quantized_output_path
        except (ImportError, Exception) as e:
            print(f"[INFO] INT8 quantization skipped: {e}")
            return onnx_output_path

    return ts_path


if __name__ == "__main__":
    export_and_quantize()
