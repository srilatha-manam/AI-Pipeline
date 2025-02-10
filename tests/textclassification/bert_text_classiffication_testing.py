import sys
import os

# Add the root of the project directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.text_classification.text_bert_classification_preprocess import fetch_data
from utils.Splitdata import split_dataset
from inference_server.textclassification.text_classification_bert_inference import predict_emotion

def test_model():
    df = fetch_data()  
    data_splits = split_dataset(df, text_columns=["processed_text"], target_column="emotion_label")
    X_test, _ = data_splits["test"]  

    sample_texts = X_test["processed_text"].head(3).tolist()  
    predictions = [predict_emotion(text) for text in sample_texts]

    assert isinstance(predictions[0], str), "Prediction should be a string"
    print("Model test passed.")

if __name__ == "__main__":
    test_model()