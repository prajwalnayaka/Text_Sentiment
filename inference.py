import torch, time
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
emotions=['fun', 'surprise', 'neutral', 'enthusiasm', 'happiness', 'hate', 'sadness', 'empty', 'love', 'relief', 'anger'] #Same order as emotions list during training
id2label = {i: label for i, label in enumerate(emotions)}
label2id = {label: i for i, label in enumerate(emotions)}
MODEL_PATH = './emotion_results/checkpoint-10636' #Version with lowest val loss
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased') #Same tokenizer used for training
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH) #The trained model
model.config.id2label = id2label #Configure the model's id2label accordingly
model.config.label2id = label2id #Configure the model's label2id accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval() #Evaluation mode
print(f"Model loaded into {device}.")
total_time = 0

def predict_emotion(text):
    global total_time
    start = time.time()
    with torch.no_grad():
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        end = time.time()
        total_time = end - start
        return model.config.id2label[predicted_class_id]

def infer(sentence):
    prediction = predict_emotion(sentence)
    result={'sentence': sentence, 'emotion': prediction, 'total_time': total_time}
    return result