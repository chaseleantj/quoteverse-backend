import cuml
from cuml.manifold.umap import UMAP

# Example usage
import numpy as np

# Generate some random data
data = np.random.rand(30, 10)

# Create UMAP instance and fit-transform data
umap_model = UMAP(n_neighbors=5, n_components=2)
embedding = umap_model.fit_transform(data)

print(embedding)


# https://stackoverflow.com/questions/56081324/why-are-google-colab-shell-commands-not-working
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# !pip install pgvector

import pandas as pd
import numpy as np
# from google.colab import drive, userdata
from tqdm.auto import tqdm
from datetime import datetime
from sqlalchemy import create_engine, text, exists
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import HALFVEC
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from openai import OpenAI
import time
import logging
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = userdata.get('DATABASE_URL')
OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
CHUNK_SIZE = 2048
MAX_ROWS = 50000
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class QuoteDB(Base):
    __tablename__ = "motivational_quotes"

    id = sa.Column(sa.Integer, primary_key=True, index=True)
    author = sa.Column(sa.String)
    book = sa.Column(sa.String)
    text = sa.Column(sa.String)
    date_created = sa.Column(sa.DateTime)
    embeddings = sa.Column(HALFVEC(EMBEDDING_DIMENSIONS))
    reduced_embeddings = sa.Column(HALFVEC(2))

def generate_embeddings(texts: List[str], client: OpenAI, retry_count: int = 0) -> Optional[List[List[float]]]:
    """Generate embeddings with retry logic"""
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
            dimensions=EMBEDDING_DIMENSIONS
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        if retry_count < MAX_RETRIES:
            logger.warning(f"Error generating embeddings: {e}. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            return generate_embeddings(texts, client, retry_count + 1)
        else:
            logger.error(f"Failed to generate embeddings after {MAX_RETRIES} attempts: {e}")
            return None

def get_existing_quotes():
    """Get set of existing quote texts to avoid duplicates"""
    with SessionLocal() as db:
        existing_quotes = db.query(QuoteDB.text).all()
        return {quote.text for quote in existing_quotes}

def process_chunk(chunk: pd.DataFrame, client: OpenAI, existing_quotes: set) -> None:
    """Process a chunk of quotes and store in database"""
    # Filter out quotes that already exist
    new_quotes = chunk[~chunk['Quote'].isin(existing_quotes)]
    if new_quotes.empty:
        return

    texts = new_quotes['Quote'].tolist()
    embeddings = generate_embeddings(texts, client)

    if embeddings is None:
        logger.error(f"Skipping chunk due to embedding generation failure")
        return

    quotes = []
    for (_, row), emb in zip(new_quotes.iterrows(), embeddings):
        quote = QuoteDB(
            author=row['Author'],
            book=row['Book'] if 'Book' in row else None,
            text=row['Quote'],
            date_created=datetime.now(),
            embeddings=emb,
            reduced_embeddings=None
        )
        quotes.append(quote)

    try:
        with SessionLocal() as db:
            db.add_all(quotes)
            db.commit()
            # Update existing_quotes set with newly added quotes
            existing_quotes.update(new_quotes['Quote'])
    except Exception as e:
        logger.error(f"Error storing quotes in database: {e}")

def fit_reducer(embeddings: List[List[float]]) -> np.ndarray:
    """Fit UMAP reducer on embeddings"""
    embeddings = np.array(embeddings)
    reducer = UMAP(
        n_neighbors=min(int(np.sqrt(len(embeddings))), len(embeddings) - 1),
        n_components=2,
    )
    return reducer.fit_transform(embeddings)

def update_reduced_embeddings(quotes, reduced_embeddings):
    """Update reduced embeddings in chunks with error handling"""
    for i in tqdm(range(0, len(quotes), CHUNK_SIZE), desc="Updating database"):
        chunk_quotes = quotes[i:i + CHUNK_SIZE]
        chunk_reduced = reduced_embeddings[i:i + CHUNK_SIZE]

        retries = 0
        while retries < MAX_RETRIES:
            try:
                with SessionLocal() as db:
                    for quote, reduced in zip(chunk_quotes, chunk_reduced):
                        quote.reduced_embeddings = reduced
                    db.add_all(chunk_quotes)
                    db.commit()
                break
            except Exception as e:
                retries += 1
                if retries == MAX_RETRIES:
                    logger.error(f"Failed to update reduced embeddings for chunk {i}: {e}")
                else:
                    logger.warning(f"Error updating reduced embeddings, retry {retries}: {e}")
                    time.sleep(RETRY_DELAY)

# Mount Google Drive
drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/Quoteverse/quotes_435k.csv'

df = pd.read_csv(file_path, nrows=MAX_ROWS)
print(len(df))
print(df.head())

df = df.dropna(subset=['Quote'])
df = df.where(pd.notnull(df), None)

print("Initializing...")
client = OpenAI(api_key=OPENAI_API_KEY)

# Get existing quotes to avoid duplicates
existing_quotes = get_existing_quotes()
print(f"Found {len(existing_quotes)} existing quotes in database")

# Read and process CSV in chunks
print("\nProcessing CSV file in chunks...")
chunk_iterator = pd.read_csv(file_path, chunksize=CHUNK_SIZE, nrows=MAX_ROWS)
total_chunks = sum(1 for _ in pd.read_csv(file_path, chunksize=CHUNK_SIZE, nrows=MAX_ROWS))

for chunk in tqdm(chunk_iterator, total=total_chunks, desc="Processing chunks"):
    chunk = chunk.dropna(subset=['Quote'])
    chunk = chunk.where(pd.notnull(chunk), None)
    if not chunk.empty:
        process_chunk(chunk, client, existing_quotes)

print("Saved embeddings to database.")

# Generate reduced embeddings only for quotes that don't have them
print("\nRetrieving quotes without reduced embeddings...")
with SessionLocal() as db:
    quotes = db.query(QuoteDB).filter(QuoteDB.reduced_embeddings.is_(None)).all()

if not quotes:
    print("No quotes need reduced embeddings")
else:
    embeddings = [quote.embeddings for quote in quotes]

    print(f"\nGenerating reduced embeddings for {len(quotes)} quotes...")
    reduced_embeddings = fit_reducer(embeddings)

    mean = np.mean(reduced_embeddings, axis=0)
    std = np.std(reduced_embeddings, axis=0)
    standardized_embeddings = (reduced_embeddings - mean) / std

    # Update quotes with reduced embeddings
    print("\nUpdating quotes with reduced embeddings...")
    update_reduced_embeddings(quotes, standardized_embeddings)

    print("\nProcessing completed!")