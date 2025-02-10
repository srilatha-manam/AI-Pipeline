import torch
from transformers import BertForSequenceClassification, BertTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
import yaml
import os
from sentence_transformers import SentenceTransformer

# Load Hugging Face API Key from config.yaml
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configurations", "config.yaml")

def load_config():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config["huggingface"]["HUGGINGFACE_TOKEN"]

class ModelManager:
    # Hold model and tokenizer instances separately for BERT and DistilBERT
    _bert_model = None
    _bert_tokenizer = None
    _distilbert_model = None
    _distilbert_tokenizer = None
    _minilm_model = None

    @staticmethod
    def load_bert_model(pretrained_model="bert-base-uncased"):
        """Load pre-trained BERT model from Hugging Face"""
        # Check if the BERT model is not already loaded
        if ModelManager._bert_model is None:
            print(f"Loading pre-trained BERT model: {pretrained_model}...")
            # Load the Hugging Face authentication token from configuration
            huggingface_token = load_config()
            # Load the pre-trained BERT model for sequence classification with 10 labels
            ModelManager._bert_model = BertForSequenceClassification.from_pretrained(
                pretrained_model, num_labels=10, token=huggingface_token
            )
            # Load the corresponding tokenizer for the pre-trained BERT model
            ModelManager._bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model, token=huggingface_token)
        
        return ModelManager._bert_model, ModelManager._bert_tokenizer

    @staticmethod
    def load_distilbert_model(pretrained_model="distilbert-base-uncased"):
        """Load pre-trained DistilBERT model from Hugging Face"""
        # Check if the DistilBERT model is not already loaded
        if ModelManager._distilbert_model is None:
            print(f"Loading pre-trained DistilBERT model: {pretrained_model}...")
            # Load the Hugging Face authentication token from configuration
            huggingface_token = load_config()
            # Load the pre-trained DistilBERT model for sequence classification with 10 labels
            ModelManager._distilbert_model = DistilBertForSequenceClassification.from_pretrained(
                pretrained_model, num_labels=10, token=huggingface_token
            )
            # Load the corresponding tokenizer for the pre-trained DistilBERT model
            ModelManager._distilbert_tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model, token=huggingface_token)
        
        return ModelManager._distilbert_model, ModelManager._distilbert_tokenizer

    @staticmethod
    def get_bert_model():
        """Return the pre-trained BERT model"""
        if ModelManager._bert_model is None:
            ModelManager.load_bert_model()
        return ModelManager._bert_model

    @staticmethod
    def get_bert_tokenizer():
        """Return the tokenizer for BERT"""
        if ModelManager._bert_tokenizer is None:
            ModelManager.load_bert_model()
        return ModelManager._bert_tokenizer

    @staticmethod
    def get_distilbert_model():
        """Return the pre-trained DistilBERT model"""
        if ModelManager._distilbert_model is None:
            ModelManager.load_distilbert_model()
        return ModelManager._distilbert_model

    @staticmethod
    def get_distilbert_tokenizer():
        """Return the tokenizer for DistilBERT"""
        if ModelManager._distilbert_tokenizer is None:
            ModelManager.load_distilbert_model()
        return ModelManager._distilbert_tokenizer
    @staticmethod
    def load_minilm_model(pretrained_model="all-MiniLM-L6-v2"):
        """Load pre-trained all-MiniLM-L6-v2 model"""
        if ModelManager._minilm_model is None:
            print(f"Loading pre-trained MiniLM model: {pretrained_model}...")
            ModelManager._minilm_model = SentenceTransformer(pretrained_model)
        
        return ModelManager._minilm_model

    @staticmethod
    def get_minilm_model():
        """Return the MiniLM model"""
        if ModelManager._minilm_model is None:
            ModelManager.load_minilm_model()
        return ModelManager._minilm_model