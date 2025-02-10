from sklearn.metrics import accuracy_score, classification_report
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference_server.textclassification.text_classification_bert_inference import predict_emotion, emotion_label_map
from utils.text_classification.text_bert_classification_preprocess import fetch_data
from utils.Splitdata import split_dataset

def evaluate_model():
    df = fetch_data()  # Fetch dataset from DB
    data_splits = split_dataset(df, text_columns=["processed_text"], target_column="emotion_label")
    
    X_test, y_test = data_splits["test"]  

    # Convert y_test text labels to numeric labels
    y_test_numeric = [list(emotion_label_map.keys())[list(emotion_label_map.values()).index(label)] for label in y_test]

    # Get model predictions
    predictions = [predict_emotion(text) for text in X_test["processed_text"].tolist()]

    # Convert predictions from string labels to numeric labels
    predictions_numeric = [list(emotion_label_map.values()).index(prediction) for prediction in predictions]

    # Calculate accuracy
    accuracy = accuracy_score(y_test_numeric, predictions_numeric)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test_numeric, predictions_numeric))

if __name__ == "__main__":
    evaluate_model()
