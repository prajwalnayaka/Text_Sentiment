import time
import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizerFast
from huggingface_hub import hf_hub_download

print("Loading Tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
print("Downloading & Loading ONNX Model...")
model_path = hf_hub_download(
    repo_id="PrajwalNayaka/text-emotion-distilbert",
    filename="model_quantized.onnx")
session = ort.InferenceSession(model_path)
emotions = ['fun', 'surprise', 'neutral', 'enthusiasm', 'happiness', 'hate', 'sadness', 'empty', 'love', 'relief', 'anger']
id2label = {i: label for i, label in enumerate(emotions)}
print("Model loaded successfully!")

def predict_emotion(text):
    start = time.time()

    # 1. Tokenize (Note: return_tensors="np" for ONNX)
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)

    # 2. Prepare Inputs for ONNX
    # ONNX Runtime expects inputs as a dictionary of numpy arrays
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }

    # 3. Run Inference
    # None means "give me all outputs"
    logits = session.run(None, ort_inputs)[0]

    # 4. Process Result
    predicted_id = np.argmax(logits, axis=1)[0]
    predicted_label = id2label[predicted_id]

    end = time.time()
    duration = end - start

    return predicted_label, duration


def infer(sentence):
    emotion, duration = predict_emotion(sentence)
    return {
        'sentence': sentence,
        'emotion': emotion,
        'total_time': round(duration, 4)
    }