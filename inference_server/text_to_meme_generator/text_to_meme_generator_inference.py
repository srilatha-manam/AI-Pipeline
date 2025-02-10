import pickle
import numpy as np
import faiss
import sys
import os
import yaml
from supabase import create_client, ClientOptions
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.model_manager import ModelManager  

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configurations/config.yaml"))

def load_config():
    try:
        with open(CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)
        return config["supabase_database"]
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

# Initialize Supabase Client
config = load_config()
if not config:
    raise Exception("Configuration loading failed")
supabase_url = config["SUPABASE_URL"]
supabase_key = config["SUPABASE_KEY"]
supabase = create_client(supabase_url, supabase_key)
options = ClientOptions(schema="dc")
supabase = create_client(supabase_url, supabase_key,options=options)
# Load MiniLM Model from ModelManager
model = ModelManager.get_minilm_model()

# Load Preprocessed Data (Pickle file containing embeddings)
def load_preprocessed_data(pkl_filename="models/processed_memes_embeddings.pkl"):
    try:
        with open(pkl_filename, 'rb') as pkl_file:
            df = pickle.load(pkl_file)
        return df
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return None

# Load FAISS Index
def load_faiss_index(index_filename="training/text_to_meme_generator/faiss_index.index"):
    try:
        index = faiss.read_index(index_filename)
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

# Find the Most Relevant Dialog
def find_most_relevant_dialog(prompt, model, faiss_index, df):
    try:
        query_embedding = model.encode(prompt, convert_to_numpy=True)
        query_embedding = np.expand_dims(query_embedding, axis=0)

        # Search FAISS index for the closest match
        distances, indices = faiss_index.search(query_embedding, 1)
        if len(indices[0]) == 0:  # No results found
            raise Exception("No results found in FAISS index.")
        
        # Extract the best match
        closest_dialog_id = df.iloc[indices[0][0]]['dialog_id']
        closest_dialog_text = df.iloc[indices[0][0]]['text']
        return closest_dialog_id, closest_dialog_text
    except Exception as e:
        print(f"Error finding most relevant dialog: {e}")
        return None, None

# Fetch meme ID and image path using dialog ID
def get_meme_info(dialog_id):
    try:
        meme_query = supabase.table('dialogs').select('meme_id').eq('dialog_id', dialog_id).execute().data
        print(meme_query)
        if not meme_query:
            raise Exception("No meme found for the dialog_id.")

        meme_id = meme_query[0]['meme_id']
        print(meme_id )
        # Fetch the corresponding image path
        image_query = supabase.table('memes_dc').select('image_path').eq('meme_id', meme_id).execute().data
        print(image_query )
        if not image_query:
            raise Exception("No image found for the meme_id.")
        return meme_id, image_query[0]['image_path']
    except Exception as e:
        print(f"Error fetching meme info: {e}")
        return None, None

# Overlay the Dialog Text on the Image
def overlay_dialog_on_image(image_path, dialog_text, image_format="image/PNG"):
    try:
        image_url = f"{supabase_url}/{image_path}"

        # Download the image
        response = requests.get(image_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download image from {image_url}, Status code: {response.status_code}")
        
        # Open the image and resize to a larger size (e.g., 400x400)
        img = Image.open(BytesIO(response.content)).resize((400, 400))  # Adjust the size of the image as needed
        draw = ImageDraw.Draw(img)
        
        # Load a custom font with a larger size (e.g., 40)
        font = ImageFont.truetype("arial.ttf", size=40)  # Adjust the font size here
        
        # Get the text size using textbbox (Bounding Box)
        bbox = draw.textbbox((0, 0), dialog_text, font=font)
        text_width = bbox[2] - bbox[0]  # Width of the bounding box
        text_height = bbox[3] - bbox[1]  # Height of the bounding box
        
        # Position the text at the center of the image or adjust accordingly
        width, height = img.size
        position = ((width - text_width) // 2, (height - text_height) // 2)  # Center the text
        
        # Draw the text on the image with green color
        draw.text(position, dialog_text, font=font, fill=(0, 255, 0))  # Green color (0, 255, 0)

        # Save the image to a byte stream
        byte_stream = BytesIO()
        img.save(byte_stream, format=image_format.upper())  # Supports PNG, JPEG, JPG
        byte_stream.seek(0)

        # Return the image in binary format
        return byte_stream
    except Exception as e:
        print(f"Error overlaying text on image: {e}")
        return None
# **Main function to connect all steps**
def generate_meme(prompt, image_format="PNG"):
    try:
        df = load_preprocessed_data()
        if df is None:
            raise Exception("Preprocessed data not loaded.")

        faiss_index = load_faiss_index()
        if faiss_index is None:
            raise Exception("FAISS index not loaded.")
        
        # Find the best-matching dialog
        dialog_id, dialog_text = find_most_relevant_dialog(prompt, model, faiss_index, df)
        if not dialog_id or not dialog_text:
            raise Exception("No relevant dialog found.")
        
        # Get meme ID and image path
        meme_id, image_path = get_meme_info(dialog_id)
        if not meme_id or not image_path:
            raise Exception("No image found for the meme.")
        
        # Generate the meme image with text overlay
        return overlay_dialog_on_image(image_path, dialog_text, image_format)
    except Exception as e:
        print(f"Error generating meme: {e}")
        return None
