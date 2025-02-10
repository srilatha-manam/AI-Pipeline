
import os
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from utils.text_classification.text_bert_classification_preprocess import fetch_data, tokenize_data
from utils.Splitdata import split_dataset
from transformers import AdamW
from utils.model_manager import ModelManager
import torch
from torch.utils.data import DataLoader, Dataset
import os
from loguru import logger

# Define dataset class
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

def train_model_bert():
    try:
        df = fetch_data()
        logger.info(f"Fetched {len(df)} records for training")

        # Split dataset
        data_splits = split_dataset(df, text_columns=["processed_text"], target_column="emotion_label")
        X_train, y_train = data_splits["train"]
        X_val, y_val = data_splits["val"]

        # Tokenize using BERT
        encodings_train = tokenize_data(X_train["processed_text"], model_type="bert")
        encodings_val = tokenize_data(X_val["processed_text"], model_type="bert")
        logger.info("Tokenization completed")

        # Convert to dataset format
        train_dataset = EmotionDataset(encodings_train, y_train.astype("category").cat.codes.tolist())
        val_dataset = EmotionDataset(encodings_val, y_val.astype("category").cat.codes.tolist())

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        # Load BERT model
        model, tokenizer = ModelManager.load_bert_model()

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=5e-5)

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Training loop
        model.train()
        for epoch in range(3):
            total_loss = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            logger.info(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

        # Save model
        save_path = "models/bert_emotion_classifier"
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"BERT model saved at {save_path}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")

if __name__ == "__main__":
    train_model_bert()
