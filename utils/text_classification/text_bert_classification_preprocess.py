import os
import sys
import yaml
import pandas as pd
from sqlalchemy import create_engine

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_manager import ModelManager  # Import ModelManager

# Load config.yaml path
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configurations/config.yaml"))

# Load DB Config from YAML
def load_db_config():
    """Load database configuration from config.yaml."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config["localdb"]["DB_CONFIG"]

# Fetch Data from PostgreSQL
def fetch_data():
    """Fetch processed text data from PostgreSQL database."""
    db_config = load_db_config()

    # Create SQLAlchemy engine
    engine = create_engine(
        f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    )

    # Use proper connection handling
    with engine.connect() as connection:
        df = pd.read_sql("SELECT id, emotion_label, processed_text FROM emotion_text_embeddings", connection)
        print("Data Retrieved from Database:")
        print(df.head())

    # Check if data is empty
    if df.empty:
        raise ValueError("Database returned an empty dataset. Check your database connection and table contents.")

    return df

# Tokenization with BERT
def tokenize_data(texts, model_type="bert"):
    """Tokenize input text using the BERT tokenizer from ModelManager."""
    tokenizer = ModelManager.get_bert_tokenizer()  # Use the tokenizer from ModelManager for BERT

    # Drop empty or NaN values before tokenization
    texts = texts.dropna()

    if texts.empty:
        raise ValueError("No valid text found for tokenization. Check database records.")

    return tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt")

if __name__ == "__main__":
    df = fetch_data()
    tokenized_texts = tokenize_data(df["processed_text"])
    print("Tokenization Complete:", tokenized_texts.keys())
