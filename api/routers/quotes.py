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
                    "text": q.text,
                    "coords": q.reduced_embeddings
                }
                for q in quotes
            ]
        }
    finally:
        db.close()

@quotes_router.post("/get-coords/")
def get_coords(
    input_strings: List[str],
    processor: EmbeddingsProcessor = Depends(get_processor)
):
    try:
        input_embeddings = processor.generate_embeddings(input_strings)
        coords = processor.transform_embeddings(input_embeddings)
        return {
            "input_strings": input_strings,
            "coords": coords.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing strings: {str(e)}")
