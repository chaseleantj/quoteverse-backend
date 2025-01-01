import os
import io
import joblib
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

from api.services.embeddings.embeddings_processor import EmbeddingsProcessor
from api.models.models import MotivationalQuote, SessionLocal, QuoteDB
from api.services.utils.utils import time_it
from api.settings import settings


@time_it
def load_quotes_from_db() -> List[MotivationalQuote]:
    with SessionLocal() as db:
        existing_quotes = db.query(QuoteDB).all()
        return [
            MotivationalQuote(
                author=q.author,
                book=q.book,
                text=q.text,
                date_created=q.date_created,
                embeddings=q.embeddings,
                reduced_embeddings=q.reduced_embeddings
            )
            for q in existing_quotes
        ]


@time_it
def load_quotes_from_csv() -> List[MotivationalQuote]:
    
    data_path = settings.DATA_PATH
    if not data_path:
        print("No data path found, skipping CSV loading. To load quotes from CSV, set the DATA_PATH environment variable.")
        return []
    
    print("Initializing quotes from CSV")
    if data_path.startswith(('http://', 'https://')):
        response = requests.get(data_path)
        df = pd.read_csv(io.StringIO(response.text), nrows=int(settings.MAX_ENTRIES))
    else:
        df = pd.read_csv(data_path, nrows=int(settings.MAX_ENTRIES))
    
    # Filter out rows where Quote is null
    df = df.dropna(subset=['Quote'])
    df = df.where(pd.notnull(df), None)
    
    return [
        MotivationalQuote(
            author=row['Author'],
            book=row['Book'],
            text=row['Quote'],
            date_created=datetime.now()
        )
        for _, row in df.iterrows()
    ]

@time_it
def save_quotes_to_db(quotes: List[MotivationalQuote]) -> None:
    print(f"Saving {len(quotes)} quotes to database")
    with SessionLocal() as db:
        db_quotes = [quote.to_db_model() for quote in quotes]
        db.add_all(db_quotes)
        db.commit()

def check_quotes_from_db() -> dict:
    """
    Check if all quotes in the database have embeddings and reduced embeddings.
    Returns a dictionary with boolean status for embeddings and reduced embeddings.
    """
    with SessionLocal() as db:
        total_quotes = db.query(QuoteDB).count()
        quotes_with_embeddings = db.query(QuoteDB).filter(QuoteDB.embeddings.is_not(None)).count()
        quotes_with_reduced = db.query(QuoteDB).filter(QuoteDB.reduced_embeddings.is_not(None)).count()
        
        return {
            "has_all_embeddings": quotes_with_embeddings == total_quotes,
            "has_all_reduced_embeddings": quotes_with_reduced == total_quotes,
            "total_quotes": total_quotes,
            "quotes_with_embeddings": quotes_with_embeddings,
            "quotes_with_reduced_embeddings": quotes_with_reduced
        }

def save_processor(processor: EmbeddingsProcessor) -> None:
    processor_path = settings.PROCESSOR_PATH
    if processor_path:
        print(f"Saving processor to {processor_path}")
        processor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(processor, processor_path)
    else:
        print("No processor path found, not saving processor.")

def load_processor() -> EmbeddingsProcessor:
    processor_path = Path(settings.PROCESSOR_PATH) if settings.PROCESSOR_PATH else None
    if processor_path and processor_path.exists():
        print(f"Loading processor from {processor_path}")
        return joblib.load(processor_path)
    else:
        print("No processor path found or no processor found at path, not loading processor.")
        return None

def process_quotes(quotes: List[MotivationalQuote]) -> Tuple[List[MotivationalQuote], EmbeddingsProcessor]:
    processor = EmbeddingsProcessor()
    # Generate embeddings and reduced embeddings
    print(f"Generating embeddings for {len(quotes)} quotes")
    quotes = processor.add_embeddings_to_quotes(quotes)
    print(f"Generating reduced embeddings for {len(quotes)} quotes")
    quotes = processor.add_reduced_embeddings_to_quotes(quotes, n_components=2)
    return quotes, processor

def init_quotes_and_processor() -> EmbeddingsProcessor:
    """Initialize quotes and embeddings processor."""
    try:
        if settings.EMPTY_DB_CONTENTS:
            # empty the database
            with SessionLocal() as db:
                db.query(QuoteDB).delete()
                db.commit()
        else:
            # Check database status
            db_status = check_quotes_from_db()
            
            if db_status["total_quotes"] > 0:
                print(f"Found {db_status['total_quotes']} existing quotes")
                
                if db_status["has_all_embeddings"] and db_status["has_all_reduced_embeddings"]:
                    print("All quotes have embeddings and reduced embeddings")
                    return EmbeddingsProcessor()
                
                # If quotes exist but missing embeddings, load and process them
                quotes = load_quotes_from_db()
                print("Processing existing quotes to add missing embeddings")
                quotes, processor = process_quotes(quotes)
                save_processor(processor)
                return processor

        # Initialize from CSV if no quotes or forced overwrite
        quotes = load_quotes_from_csv()
        if len(quotes) == 0:
            print("No quotes found, skipping processing")
            return EmbeddingsProcessor()
        
        quotes, processor = process_quotes(quotes)
        save_quotes_to_db(quotes)
        save_processor(processor)
        return processor

    except Exception as e:
        print(f"Error initializing quotes: {str(e)}")
        raise