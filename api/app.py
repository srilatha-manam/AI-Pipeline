from fastapi import FastAPI, Query
from inference_server.textclassification.text_classification_bert_inference import predict_emotion
from inference_server.textclassification.text_classification_dislitbert_inference import predict_emotion_distilbert
from inference_server.text_to_meme_generator.text_to_meme_generator_inference import generate_meme
from data.text_prompt import PromptRequest, PromptResponse, ImageResponse
from loguru import logger
from logs.log_file import configure_logger
from fastapi.responses import Response

app = FastAPI()
logger = configure_logger()

@app.post("/bert-classification-predict", response_model=PromptResponse)
def classify_text(request: PromptRequest):
    try:       
        text = request.text
        logger.info(f"Received text: {text}")  # Log received text
        
        # Get the emotion label from the predict_emotion function
        prediction = predict_emotion(text)
        
        logger.info(f"Prediction: {prediction}")  # Log prediction result
        
        return PromptResponse(emotion_label=prediction)
    
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")  # Log any exception
        return {"error": "Internal server error"}
    
@app.post("/dislitbert-classification-predict", response_model=PromptResponse)
def classify_text(request: PromptRequest):
    try:       
        text = request.text
        logger.info(f"Received text: {text}")  # Log received text
        
        # Get the emotion label from the predict_emotion function
        prediction = predict_emotion_distilbert(text)
        
        logger.info(f"Prediction: {prediction}")  # Log prediction result
        
        return PromptResponse(emotion_label=prediction)
    
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")  # Log any exception
        return {"error": "Internal server error"}
    
@app.post("/generate-meme")
async def generate_meme_api(request: PromptRequest, image_format: str = Query("PNG", enum=["PNG", "JPG", "JPEG"])):
    try:
        logger.info(f"Generating meme for: {request.text}")
        # Generate the meme image (BytesIO object)
        meme_image = generate_meme(request.text, image_format)
        print(meme_image)
        # Return the image in binary format
        return Response(content=meme_image.getvalue(), media_type=f"image/{image_format.lower()}")
    except Exception as e:
        logger.error(f"Error generating meme: {str(e)}")  
        return {"error": "Internal server error"}
    