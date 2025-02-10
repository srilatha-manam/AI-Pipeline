from pydantic import BaseModel
from typing import Optional

class PromptRequest(BaseModel):
    text: str  # The input text prompt

class PromptResponse(BaseModel):
    emotion_label: str

class ImageResponse(BaseModel):   
    meme_image: Optional[str] = "image/png"
      