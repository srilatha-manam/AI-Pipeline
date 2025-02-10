import numpy as np
import faiss
import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load Sentence-BERT Model (384-dimensional)
def load_sentence_bert_model():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Outputs 384-dimensional embeddings
    return model

# Get Embedding from Sentence-BERT
def get_embedding(text, model):
    embedding = model.encode(text)
    print(f"Generated embedding shape: {embedding.shape}")  # Debugging output
    return embedding

# Load Dialogs and Embeddings from Pickle File
def get_dialogs_from_pkl(pkl_file="models/processed_memes_embeddings.pkl"):
    try:
        with open(pkl_file, "rb") as file:
            df = pickle.load(file)  # Load as Pandas DataFrame

        print(f"Loaded {len(df)} dialogs from {pkl_file}")
        print("Example dialog structure:\n", df.head())  # Print first few rows

        # Ensure required columns exist
        if "embedding" not in df.columns or "dialog_id" not in df.columns or "text" not in df.columns:
            print("Error: Pickle file does not contain required columns!")
            return []

        # Convert DataFrame to List of Dictionaries
        dialogs = df.to_dict(orient="records")  # Ensures correct structure

        return dialogs
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return []

# Load FAISS Index
def load_faiss_index(index_filename="training/text_to_meme_generator/faiss_index.index"):
    try:
        index = faiss.read_index(index_filename)
        print(f"FAISS index loaded successfully with dimension: {index.d}")
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

# Find Most Relevant Dialog Using Cosine Similarity
def find_most_relevant_dialog_cosine(prompt, model, dialogs):
    prompt_embedding = get_embedding(prompt, model)
    similarities = []

    for dialog in dialogs:
        if 'embedding' not in dialog:
            print(f"Skipping dialog without embedding: {dialog}")
            continue

        dialog_embedding = np.array(dialog['embedding'])  # Extract from DataFrame
        sim = cosine_similarity([prompt_embedding], [dialog_embedding])[0][0]
        similarities.append((dialog['dialog_id'], dialog['text'], sim))
    
    if not similarities:
        return None  # Return None if no valid embeddings found
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[0]  # Return the most similar dialog

# Evaluate Model Performance for Both FAISS and Cosine Similarity
def evaluate_model(test_data, faiss_index, dialogs, model):
    true_labels = []
    predicted_labels_faiss = []
    predicted_labels_cosine = []

    for prompt, actual_dialog_text in test_data:
        query_embedding = get_embedding(prompt, model).reshape(1, -1)

        if query_embedding.shape[1] != faiss_index.d:
            print(f"Error: FAISS index dimension ({faiss_index.d}) does not match query embedding dimension ({query_embedding.shape[1]}).")
            return
        
        # FAISS Search
        distances, indices = faiss_index.search(query_embedding, 1)

        # Ensure a valid index is returned for FAISS
        if indices[0][0] == -1 or indices[0][0] >= len(dialogs):
            print(f"No match found in FAISS for prompt: {prompt}")
            predicted_labels_faiss.append("Unknown")
        else:
            predicted_dialog_text_faiss = dialogs[indices[0][0]]['text']
            predicted_labels_faiss.append(predicted_dialog_text_faiss)

        # Cosine Similarity Search
        predicted_dialog_cosine = find_most_relevant_dialog_cosine(prompt, model, dialogs)
        if predicted_dialog_cosine is None:
            predicted_labels_cosine.append("Unknown")
        else:
            predicted_labels_cosine.append(predicted_dialog_cosine[1])

        true_labels.append(actual_dialog_text)

        print(f"\nPrompt: {prompt}")
        print(f"Actual: {actual_dialog_text}")
        print(f"FAISS Predicted: {predicted_labels_faiss[-1]}")
        print(f"Cosine Predicted: {predicted_labels_cosine[-1]}")
    
    # Compute Precision, Recall, F1 Score for FAISS
    precision_faiss = precision_score(true_labels, predicted_labels_faiss, average='macro', zero_division=0)
    recall_faiss = recall_score(true_labels, predicted_labels_faiss, average='macro', zero_division=0)
    f1_faiss = f1_score(true_labels, predicted_labels_faiss, average='macro', zero_division=0)

    # Compute Precision, Recall, F1 Score for Cosine Similarity
    precision_cosine = precision_score(true_labels, predicted_labels_cosine, average='macro', zero_division=0)
    recall_cosine = recall_score(true_labels, predicted_labels_cosine, average='macro', zero_division=0)
    f1_cosine = f1_score(true_labels, predicted_labels_cosine, average='macro', zero_division=0)

    print("\n🔹 Model Performance Results:")
    print(f"FAISS - Precision: {precision_faiss}, Recall: {recall_faiss}, F1-Score: {f1_faiss}")
    print(f"Cosine - Precision: {precision_cosine}, Recall: {recall_cosine}, F1-Score: {f1_cosine}")

# Run Model Evaluation
if __name__ == "__main__":
    dialogs = get_dialogs_from_pkl()
    model = load_sentence_bert_model()
    faiss_index = load_faiss_index()

    if faiss_index is None:
        print("Error: FAISS index not loaded. Exiting...")
        exit(1)

    test_data = [
        ("ela", "appude ela aypo kottav ra"), 
        ("chepu", "chepu"),
        ("hor", "horizon ee ga"),
        ("sir", "sir mana college lo mi antha handsome professor evaru leru sir"),
        ("dad", "Dad: Fail ayii natalu veyaku!")
    ]
    
    evaluate_model(test_data, faiss_index, dialogs, model)
