import numpy as np
import faiss
import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from difflib import SequenceMatcher
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

#  Load Sentence-BERT Model (384-dimensional embeddings)
def load_sentence_bert_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-d embeddings

#  Load DPR Model (Dense Passage Retrieval)
def load_dpr_model():
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    return model, tokenizer

#  Get Embedding for Sentence-BERT & DPR
def get_embedding(text, model, tokenizer=None, model_type="sentence-bert"):
    if model_type == "sentence-bert":
        embedding = model.encode(text)
    elif model_type == "dpr":
        inputs = tokenizer(text, return_tensors="pt")
        embedding = model(**inputs).pooler_output.detach().numpy().squeeze()
    else:
        raise ValueError("Invalid model type.")
    return embedding

#  Load Dialogs and Embeddings from Pickle File
def get_dialogs_from_pkl(pkl_file="models/processed_memes_embeddings.pkl"):
    try:
        with open(pkl_file, "rb") as file:
            df = pickle.load(file)

        print(f"Loaded {len(df)} dialogs from {pkl_file}")
        print("Example dialog structure:\n", df.head())

        if "embedding" not in df.columns or "dialog_id" not in df.columns or "text" not in df.columns:
            print("Error: Pickle file does not contain required columns!")
            return []

        return df.to_dict(orient="records")  # Convert to list of dictionaries
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return []

#  Load FAISS Index
def load_faiss_index(index_filename="training/text_to_meme_generator/faiss_index.index"):
    try:
        index = faiss.read_index(index_filename)
        print(f"FAISS index loaded successfully with dimension: {index.d}")
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

#  Prepare BM25 Model for Lexical Matching
def prepare_bm25(dialogs):
    corpus = [dialog["text"].lower().split() for dialog in dialogs]
    return BM25Okapi(corpus), corpus

#  Find Most Relevant Dialog Using BM25
def find_most_relevant_dialog_bm25(prompt, bm25, corpus, dialogs):
    scores = bm25.get_scores(prompt.lower().split())
    best_index = np.argmax(scores)
    return dialogs[best_index]["dialog_id"], dialogs[best_index]["text"]

#  Find Most Relevant Dialog Using Cosine Similarity
def find_most_relevant_dialog_cosine(prompt, model, dialogs):
    prompt_embedding = get_embedding(prompt, model)
    similarities = [
        (dialog["dialog_id"], dialog["text"], cosine_similarity([prompt_embedding], [np.array(dialog["embedding"])])[0][0])
        for dialog in dialogs if "embedding" in dialog
    ]
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[0] if similarities else None

# Function to Calculate Partial Match Accuracy
def calculate_fuzzy_accuracy(true_labels, predicted_labels):
    similarities = [
        SequenceMatcher(None, true, pred).ratio() for true, pred in zip(true_labels, predicted_labels)
    ]
    
    # Convert similarity scores into binary match (1 = match, 0 = no match)
    threshold = 0.6  # Adjust threshold based on need
    matches = [1 if sim >= threshold else 0 for sim in similarities]
    
    accuracy = sum(matches) / len(matches) if len(matches) > 0 else 0
    return accuracy

#  Evaluate Accuracy for All Models
def evaluate_models(test_data, faiss_index, dialogs, sentence_bert_model, dpr_model, dpr_tokenizer, bm25, corpus):
    true_labels = []
    faiss_predictions = []
    cosine_predictions = []
    bm25_predictions = []
    dpr_predictions = []

    for prompt, actual_dialog_text in test_data:
        query_embedding = get_embedding(prompt, sentence_bert_model).reshape(1, -1)

        #  FAISS Search
        if query_embedding.shape[1] == faiss_index.d:
            distances, indices = faiss_index.search(query_embedding, 1)
            faiss_pred = dialogs[indices[0][0]]["text"] if indices[0][0] != -1 else "Unknown"
        else:
            faiss_pred = "Unknown"
        faiss_predictions.append(faiss_pred)

        # Cosine Similarity Search
        cosine_result = find_most_relevant_dialog_cosine(prompt, sentence_bert_model, dialogs)
        cosine_pred = cosine_result[1] if cosine_result else "Unknown"
        cosine_predictions.append(cosine_pred)

        #  BM25 Search
        bm25_id, bm25_text = find_most_relevant_dialog_bm25(prompt, bm25, corpus, dialogs)
        bm25_predictions.append(bm25_text)

        #  DPR Search
        dpr_embedding = get_embedding(prompt, dpr_model, dpr_tokenizer, "dpr").reshape(1, -1)
        distances, indices = faiss_index.search(dpr_embedding, 1)
        dpr_pred = dialogs[indices[0][0]]["text"] if indices[0][0] != -1 else "Unknown"
        dpr_predictions.append(dpr_pred)

        true_labels.append(actual_dialog_text)

    #  Compute Accuracy Metrics for Each Model
    models = ["FAISS", "Cosine Similarity", "BM25", "DPR"]
    predictions = [faiss_predictions, cosine_predictions, bm25_predictions, dpr_predictions]

    for model_name, pred in zip(models, predictions):
        fuzzy_accuracy = calculate_fuzzy_accuracy(true_labels, pred)
        precision = precision_score(true_labels, pred, average='macro', zero_division=0)
        recall = recall_score(true_labels, pred, average='macro', zero_division=0)
        f1 = f1_score(true_labels, pred, average='macro', zero_division=0)

        print(f"\n🔹 {model_name} Performance:")
        print(f"🔹 Precision: {precision:.3f}")
        print(f"🔹 Recall: {recall:.3f}")
        print(f"🔹 F1-Score: {f1:.3f}")
        print(f"🔹 Fuzzy Accuracy: {fuzzy_accuracy:.3f}")

#  Run Evaluation
if __name__ == "__main__":
    dialogs = get_dialogs_from_pkl()
    faiss_index = load_faiss_index()
    sentence_bert_model = load_sentence_bert_model()
    dpr_model, dpr_tokenizer = load_dpr_model()
    bm25, corpus = prepare_bm25(dialogs)

    if faiss_index is None:
        print("Error: FAISS index not loaded. Exiting...")
        exit(1)

    test_data = [
        ("client", "My friends at Office Party"), 
        ("happy customer", "This is the happiest moment of my life"),
        ("sir", "sir mana college lo mi antha handsome professor evaru leru sir"),
        ("dad", "Dad: Fail ayii natalu veyaku!"),
        ("chepu", "chepu")
    ]

    evaluate_models(test_data, faiss_index, dialogs, sentence_bert_model, dpr_model, dpr_tokenizer, bm25, corpus)
