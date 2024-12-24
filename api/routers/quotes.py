from typing import List
from fastapi import APIRouter, Depends, HTTPException
from api.models.models import SessionLocal, QuoteDB
from api.dependencies import get_processor
from api.services.embeddings.embeddings_processor import EmbeddingsProcessor


quotes_router = APIRouter(prefix="/quotes")

@quotes_router.get("/")
def get_quotes(count: int = 100):
    db = SessionLocal()
    try:
        quotes = db.query(QuoteDB).limit(count).all()
        return {
            "count": len(quotes),
            "quotes": [
                {
                    "author": q.author,
                    "quote": q.quote,
                    "pca_coords": q.pca_embeddings
                }
                for q in quotes
            ]
        }
    finally:
        db.close()

@quotes_router.post("/pca-coordinates/")
def get_pca_coordinates(
    input_strings: List[str],
    processor: EmbeddingsProcessor = Depends(get_processor)
):
    try:
        input_embeddings = processor.generate_embeddings(input_strings)
        pca_coords = processor.transform_embeddings(input_embeddings)
        return {
            "input_strings": input_strings,
            "pca_coordinates": pca_coords.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing strings: {str(e)}")
