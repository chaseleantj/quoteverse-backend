from typing import List, Literal
from sqlalchemy.sql import func
from fastapi import APIRouter, Depends, HTTPException

from api.models.models import SessionLocal, QuoteDB
from api.dependencies import get_processor
from api.services.embeddings.embeddings_processor import EmbeddingsProcessor
from api.services.search.similarity_search import similarity_search
from api.services.search.text_search import text_search


quotes_router = APIRouter(prefix="/quotes")

@quotes_router.get("/")
def get_quotes(count: int = 100, randomize: bool = False):
    db = SessionLocal()
    try:
        if randomize:
            quotes = db.query(QuoteDB).order_by(func.random()).limit(count).all()
        else:
            quotes = db.query(QuoteDB).limit(count).all()
        
        return {
            "status": "success",
            "data": {
                "quotes": [
                    {
                        "id": q.id,
                        "author": q.author,
                        "book": q.book,
                        "text": q.text,
                        "coords": q.reduced_embeddings.to_list()
                    }
                    for q in quotes
                ]
            },
            "metadata": {
                "count": len(quotes),
                "limit": count
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing strings: {str(e)}")
    finally:
        db.close()
    
    
@quotes_router.post("/get-similar-quotes/")
def get_similar_quotes(
    input_strings: List[str],
    k: int = 5,
    max_distance: float = None,
    processor: EmbeddingsProcessor = Depends(get_processor)
):
    try:
        results = similarity_search(input_strings, k, max_distance, processor)
        return {
            "status": "success",
            "data": {
                "queries": [
                    {
                        "input_text": input_text,
                        "similar_quotes": [
                            {
                                "id": quote.id,
                                "author": quote.author,
                                "book": quote.book,
                                "text": quote.text,
                                "coords": quote.reduced_embeddings.to_list(),
                                "distance": distance
                            }
                            for quote, distance in quotes
                        ]
                    }
                    for input_text, quotes in zip(input_strings, results)
                ]
            },
            "metadata": {
                "query_count": len(input_strings),
                "results_per_query": k,
                "max_distance": max_distance
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing strings: {str(e)}")


@quotes_router.post("/search-quotes/")
def search_quotes(
    search_texts: List[str],
    search_by: Literal["author", "book"],
    k: int = 5,
    strict: bool = False,
):
    try:
        results = text_search(search_texts, search_by, k, strict)
        return {
            "status": "success",
            "data": {
                "queries": [
                    {
                        "input_text": text,
                        "quotes": [
                            {
                                "id": quote.id,
                                "author": quote.author,
                                "book": quote.book,
                                "text": quote.text,
                                "coords": quote.reduced_embeddings.to_list(),
                            }
                            for quote in quotes
                        ]
                    }
                    for text, quotes in zip(search_texts, results)
                ]
            },
            "metadata": {
                "query_count": len(search_texts),
                "results_per_query": k,
                "search_by": search_by,
                "strict": strict
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing strings: {str(e)}")
