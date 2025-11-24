import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. Load your trained model
MODEL_PATH = './emotion_results/checkpoint-10636'  # Point to your best local checkpoint
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# 2. Put model in CPU mode for export
model.cpu()
model.eval()

# 3. Create dummy input (needed for the conversion tracer)
dummy_text = "This is a test sentence"
inputs = tokenizer(dummy_text, return_tensors="pt")

# 4. Export to ONNX
output_path = "model.onnx"
print(f"Converting model to {output_path}...")

torch.onnx.export(
    model,
    (inputs['input_ids'], inputs['attention_mask']),
    output_path,
    export_params=True,                         # Store weights inside the file
    opset_version=14,                           # ONNX version
    do_constant_folding=True,                   # Optimization
    input_names=['input_ids', 'attention_mask'],# Name the inputs
    output_names=['logits'],                    # Name the output
    dynamic_axes={                              # Allow variable length sentences
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'}
    }
)
print("Success! Model saved as model.onnx")
quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QUInt8  # Convert weights to 8-bit integers
)
print("Done! Saved as model_quantized.onnx")