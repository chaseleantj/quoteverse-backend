import os
import pandas as pd
from datetime import datetime
from api.services.embeddings.embeddings_processor import EmbeddingsProcessor
from api.models.models import MotivationalQuote, SessionLocal, QuoteDB


def init_quotes_and_processor() -> EmbeddingsProcessor:
    """
    Initialize quotes from CSV file with embeddings and load into database.
    Returns the fitted processor.
    """
    db = SessionLocal()
    processor = EmbeddingsProcessor()
    
    try:
        # Check if quotes already exist
        existing_count = db.query(QuoteDB).count()
        if existing_count > 0:
            print(f"Found {existing_count} existing quotes, skipping initialization")
            # Still need to fit the processor with existing quotes
            existing_quotes = db.query(QuoteDB).all()
            quotes = [
                MotivationalQuote(
                    author=q.author,
                    text=q.text,
                    date_created=q.date_created,
                    embeddings=q.embeddings,
                    reduced_embeddings=q.reduced_embeddings
                )
                for q in existing_quotes
            ]
            processor.add_reduced_embeddings_to_quotes(quotes)
            return processor
            
        # Read quotes from CSV
        df = pd.read_csv(os.getenv('DATA_PATH'))
        print(f"Loaded {len(df)} quotes from {os.getenv('DATA_PATH')}")

        # Replace falsy values with None
        df = df.where(pd.notnull(df), None)

        # Convert to MotivationalQuote objects
        quotes = [
            MotivationalQuote(
                author=row['Author'],
                text=row['Quote'],
                date_created=datetime.now()
            )
            for _, row in df.iterrows()
        ]
        
        # Generate embeddings
        quotes_with_embeddings = processor.process_quotes(quotes)
        print(f"Generated embeddings for {len(quotes_with_embeddings)} quotes")
        
        # Add reduced embeddings
        quotes_with_reduced_embeddings = processor.add_reduced_embeddings_to_quotes(quotes_with_embeddings, n_components=2)
        print(f"Added reduced embeddings for {len(quotes_with_reduced_embeddings)} quotes")
        
        # Store in database
        try:
            # Convert to DB models and add to session
            db_quotes = [quote.to_db_model() for quote in quotes_with_reduced_embeddings]
            db.add_all(db_quotes)
            db.commit()
            print(f"Successfully loaded {len(db_quotes)} quotes into database")
            return processor
        finally:
            db.close()
            
    except Exception as e:
        print(f"Error initializing quotes: {str(e)}")
        raise