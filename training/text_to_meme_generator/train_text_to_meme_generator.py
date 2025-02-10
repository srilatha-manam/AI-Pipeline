
#traing model using FAISS index no need to use traing data set fro splitdata.py
import pickle
import numpy as np
import faiss
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load preprocessed data (pickle file containing embeddings)
def load_preprocessed_data(pkl_filename="models/processed_memes_embeddings.pkl"):
    with open(pkl_filename, 'rb') as pkl_file:
        df = pickle.load(pkl_file)
    return df

# FAISS setup
def setup_faiss_index(embeddings):
    d = embeddings.shape[1]  # Dimension of the embeddings (MiniLM produces 384-dimensional embeddings)
    index = faiss.IndexFlatL2(d)  # L2 distance for similarity
    index.add(embeddings)  # Add embeddings to FAISS index
    return index

# Create FAISS index and save it
def create_faiss_index():
    df = load_preprocessed_data()

    # Create FAISS index
    embeddings = np.array([embedding for embedding in df['embedding']])
    faiss_index = setup_faiss_index(embeddings)
    faiss_index_path = os.path.join(os.path.dirname(__file__), 'faiss_index.index')
    # Save FAISS index to disk
    faiss.write_index(faiss_index, faiss_index_path)

    print("FAISS index created and saved successfully!")

# Run the FAISS index creation
if __name__ == "__main__":
    create_faiss_index()
