#it uses fine tuned model for text classification predection 

from transformers import BertForSequenceClassification, BertTokenizer
import torch
from loguru import logger

# Load the fine-tuned model and tokenizer
model_path = "models/bert_emotion_classifier"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# static numeric label for emotion
emotion_label_map = {
    0: "angry",
    1: "happy",
    2: "sad",
    3: "fearful",
    4: "surprised",
    5: "confused",
    6: "wonder",   
    7: "heroism",
    8: "disgust",
    9: "neutral"  
}

def predict_emotion(text):
    try:
        # Log the input text (request)
        logger.info(f"Predicting emotion for text: {text}")

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()  # Get predicted class
        
        # Map prediction to label
        emotion_label = emotion_label_map.get(prediction, "unknown")  # Default to "unknown" if not found
        
        # Log the predicted emotion label
        logger.info(f"Predicted emotion: {emotion_label}")
        return emotion_label

    except Exception as e:
        # Log any errors that occur during prediction
        logger.error(f"Error predicting emotion for text: {text}. Error: {str(e)}")
        return "error"

