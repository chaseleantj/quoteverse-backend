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

@time_it
def load_quotes_from_db() -> List[MotivationalQuote]:
    with SessionLocal() as db:
        existing_quotes = db.query(QuoteDB).all()
        return [
            MotivationalQuote(
                author=q.author,
                text=q.text,
                date_created=q.date_created,
                embeddings=q.embeddings,
                reduced_embeddings=q.reduced_embeddings
            )
            for q in existing_quotes
        ]


@time_it
def load_quotes_from_csv() -> List[MotivationalQuote]:
    print("Initializing quotes from CSV")
    data_path = os.getenv('DATA_PATH')
    
    # Handle URL or local file
    if data_path.startswith(('http://', 'https://')):
        response = requests.get(data_path)
        df = pd.read_csv(io.StringIO(response.text), nrows=int(os.getenv('MAX_ENTRIES')))
    else:
        df = pd.read_csv(data_path, nrows=int(os.getenv('MAX_ENTRIES')))
    
    # Filter out rows where Quote is null
    df = df.dropna(subset=['Quote'])
    df = df.where(pd.notnull(df), None)
    
    return [
        MotivationalQuote(
            author=row['Author'],
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

def get_processor_path() -> Path:
    return Path(os.getenv('MODEL_PATH')) / 'embeddings_processor.joblib'

@time_it
def save_processor(processor: EmbeddingsProcessor) -> None:
    processor_path = get_processor_path()
    print(f"Saving processor to {processor_path}")
    processor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(processor, processor_path)

@time_it
def load_processor() -> EmbeddingsProcessor:
    processor_path = get_processor_path()
    return joblib.load(processor_path)

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
        
        if os.getenv('FORCE_OVERWRITE_DB') == 'true':
            # empty the database
            db = SessionLocal()
            db.query(QuoteDB).delete()
            db.commit()
            db.close()
            existing_count = 0
        else:
            # Check for existing quotes
            db = SessionLocal()
            existing_count = db.query(QuoteDB).count()
            db.close()

            if existing_count > 0:
                print(f"Found {existing_count} existing quotes")
                quotes = load_quotes_from_db()
                
                # Try to load existing processor
                processor_path = get_processor_path()
                if processor_path.exists():
                    print("Loading existing processor")
                    return load_processor()
                
                # If no processor exists, create and fit a new one
                print("Creating new processor with existing quotes")
                quotes, processor = process_quotes(quotes)
                save_processor(processor)
                return processor

        # Initialize from CSV
        quotes = load_quotes_from_csv()
        quotes, processor = process_quotes(quotes)
        # Save to database and save processor
        save_quotes_to_db(quotes)
        save_processor(processor)
        
        return processor

    except Exception as e:
        print(f"Error initializing quotes: {str(e)}")
        raise