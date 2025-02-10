#preprocessing data

import pandas as pd
from supabase import create_client, ClientOptions
import yaml
import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_manager import ModelManager  
import pickle

# Load Supabase Config
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configurations/config.yaml"))

def load_config():
     try:
        with open(CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)
        return config["supabase_database"]
     except Exception as e:        
        return e

# Initialize Supabase Client with schema
config = load_config()
supabase_url = config["SUPABASE_URL"]
supabase_key = config["SUPABASE_KEY"]
options = ClientOptions(schema="dc")
supabase = create_client(supabase_url, supabase_key,options=options)

# Load MiniLM Model from ModelManager
model = ModelManager.get_minilm_model()
def preprocess_and_store_pkl():
    try:

        print("Fetching raw meme data from Supabase (schema: dc)...")
        
        # Fetch data from `dialogs` table in `dc` schema
        response = supabase.table("dialogs").select("*").execute()
        if not response.data:
            print("No data found in 'dialogs' table.")
            return

        df = pd.DataFrame(response.data)
        
        # Ensure 'text' column exists
        if "text" not in df.columns:
            print("'text' column missing in dataset!")
            return

        # Preprocess: Remove duplicates and missing values
        df = df.drop_duplicates(subset=["text"]).dropna(subset=["text"])

        # Clean the 'text' column )
        df["cleaned_text"] = df["text"].apply(lambda x: x.strip().lower())  #  strip and lowercase

        # Convert text to embeddings using MiniLM
        print("Generating embeddings for meme texts...")
        df["embedding"] = df["cleaned_text"].apply(lambda x: model.encode(x).tolist())

        models_folder = os.path.join(os.path.dirname(__file__), '../../models')
        os.makedirs(models_folder, exist_ok=True)

        # Save preprocessed data and embeddings to a pickle file in the models folder
        pkl_filename = os.path.join(models_folder, "processed_memes_embeddings.pkl")
        print(f"Saving preprocessed data and embeddings to {pkl_filename}...")
        with open(pkl_filename, 'wb') as pkl_file:
            pickle.dump(df, pkl_file)
        
        print("Data preprocessing and pickle file storage completed!")
    except Exception as e:
        return e
# Run Preprocessing Manually (for testing)
if __name__ == "__main__":
    preprocess_and_store_pkl()

#embedding text saved as pkl file  
#is this good to run this code on regular intravels