# LLM Platform Embeddings

A simple FastAPI/Transformers Embeddings implementation mimics the [Jina AI Embeddings API](https://jina.ai/embeddings).

Defaults to [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3). Can be overwritten using `MODEL` environment variable.

```bash
# Using Embeddings Model
docker run -it --rm -p 8000:8000 -e MODEL=jinaai/jina-embeddings-v3 ghcr.io/adrianliechti/llama-embeddings

# Using CLIP Model
docker run -it --rm -p 8000:8000 -e MODEL=jinaai/jina-clip-v1 ghcr.io/adrianliechti/llama-embeddings
```

## Text Embeddings

Using [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
        "input": [
          "A blue cat",
          "A red dog",          
          {"text": "btw to represent image u can either use URL or encode image into base64 like below."}
        ]
      }'
```

```json
{
    "object": "list",
    "model": "jinaai/jina-embeddings-v3",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [
                -0.008155452087521553,
                0.004017454106360674,
                ...
            ]
        },
        {
            "object": "embedding",
            "index": 1,
            "embedding": [
                -0.03279036656022072,
                0.005651661194860935,
                ...
            ]
        },
        {
            "object": "embedding",
            "index": 2,
            "embedding": [
                -0.04352213442325592,
                0.014829625375568867,
                ...
            ]
        }
    ]
}
```

## Mixed Embeddings

Using [jinaai/jina-clip-v1](https://huggingface.co/jinaai/jina-clip-v1)

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
        "input": [
          "A blue cat",
          "A red dog",          
          {"text": "btw to represent image u can either use URL or encode image into base64 like below."}, 
          {"image": "https://i.pinimg.com/600x315/21/48/7e/21487e8e0970dd366dafaed6ab25d8d8.jpg"}, 
          {"image": "R0lGODlhEAAQAMQAAORHHOVSKudfOulrSOp3WOyDZu6QdvCchPGolfO0o/XBs/fNwfjZ0frl3/zy7////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAkAABAALAAAAAAQABAAAAVVICSOZGlCQAosJ6mu7fiyZeKqNKToQGDsM8hBADgUXoGAiqhSvp5QAnQKGIgUhwFUYLCVDFCrKUE1lBavAViFIDlTImbKC5Gm2hB0SlBCBMQiB0UjIQA7"}
        ]
      }'
```

```json
{
    "object": "list",
    "model": "jinaai/jina-clip-v1",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [
                -0.008155452087521553,
                0.004017454106360674,
                ...
            ]
        },
        {
            "object": "embedding",
            "index": 1,
            "embedding": [
                -0.03279036656022072,
                0.005651661194860935,
                ...
            ]
        },
        {
            "object": "embedding",
            "index": 2,
            "embedding": [
                -0.04352213442325592,
                0.014829625375568867,
                ...
            ]
        },
        {
            "object": "embedding",
            "index": 3,
            "embedding": [
                -0.02155349962413311,
                0.010659383609890938,
                ...
            ]
        },
        {
            "object": "embedding",
            "index": 4,
            "embedding": [
                -0.04216967895627022,
                0.017293065786361694,
                ...
            ]
        }
    ]
}
```