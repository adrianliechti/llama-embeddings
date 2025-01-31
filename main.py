import os
import io
import base64
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from typing import List, Optional

from PIL import Image
from sentence_transformers import SentenceTransformer

model_name = os.getenv("MODEL", "jinaai/jina-embeddings-v3")
model = SentenceTransformer(model_name, trust_remote_code=True)

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
def embed(request: EmbeddingRequest):
    input = request.input

    data = []
    
    for i, input in enumerate(input):
        if input.text:
            embedding = model.encode(input.text, normalize_embeddings=True)

            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding.tolist()
            })
            
        if input.image:
            if input.image.startswith("http://") or input.image.startswith("https://"):
                embedding = model.encode(input.image, normalize_embeddings=True)
            else:
                image_data = base64.b64decode(input.image)

                image = Image.open(io.BytesIO(image_data))
                image = image.convert("RGB")

                embedding = model.encode(image, normalize_embeddings=True)
            
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding.tolist()
            })

    return {
        "object": "list",
        "model": model_name,
        "data": data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)