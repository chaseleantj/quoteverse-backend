from fastapi import Request
from api.services.embeddings.embeddings_processor import EmbeddingsProcessor

async def get_processor(request: Request) -> EmbeddingsProcessor:
    if not hasattr(request.app.state, "processor"):
        raise RuntimeError("Processor not initialized. Server startup may have failed.")
    return request.app.state.processor
