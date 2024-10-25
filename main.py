import io
import base64
import requests
import uvicorn

from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel, field_validator

from PIL import Image
from fastembed import ImageEmbedding, TextEmbedding

text_embedding = TextEmbedding(model_name="intfloat/multilingual-e5-large")
image_embedding = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")

app = FastAPI(
    title="LLM Platform Embeddings"
)

class InputItem(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None

class EmbeddingRequest(BaseModel):
    input: List[InputItem]

    @field_validator('input', mode='before')
    def parse_input(cls, v):
        if isinstance(v, str):
            v = [v]
        elif not isinstance(v, list):
            raise ValueError('Input must be a string or a list')
        
        input = []

        for item in v:
            if isinstance(item, str):
                input.append({'text': item})
            elif isinstance(item, dict):
                input.append(item)
            else:
                raise ValueError('Each item in input must be a string or a dictionary')
            
        return input

@app.post("/embeddings")
@app.post("/v1/embeddings")
async def embed(request: EmbeddingRequest):
    data = []
    
    for i, input in enumerate(request.input):
        if input.text:
            result = list(text_embedding.embed(input.text))
            embedding = result[0].tolist()

            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding,
            })
            
        if input.image:
            if input.image.startswith("http://") or input.image.startswith("https://"):
               response = requests.get(input.image)
               response.raise_for_status()

               image = Image.open(io.BytesIO(response.content))
               image = image.convert("RGB")
               
            else:
                image_data = base64.b64decode(input.image)

                image = Image.open(io.BytesIO(image_data))
                image = image.convert("RGB")

            
            result = list(image_embedding.embed(image))
            embedding = result[0].tolist()
            
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding
            })
    
    return {
        "object": "list",
        "model": "default",
        "data": data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)