import os

from datetime import datetime
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ARRAY, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel

# SQLAlchemy setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Model
class QuoteDB(Base):
    __tablename__ = "motivational_quotes"

    id = Column(Integer, primary_key=True, index=True)
    author = Column(String)
    quote = Column(String)
    date_created = Column(DateTime)
    embeddings = Column(ARRAY(Float))  # Store embeddings as array
    pca_embeddings = Column(ARRAY(Float))  # Store PCA embeddings as array

# Pydantic Model
class MotivationalQuote(BaseModel):
    author: str
    quote: str
    date_created: datetime
    embeddings: Optional[List[float]] = None
    pca_embeddings: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True

    def to_db_model(self) -> QuoteDB:
        return QuoteDB(
            author=self.author,
            quote=self.quote,
            date_created=self.date_created,
            embeddings=self.embeddings,
            pca_embeddings=self.pca_embeddings
        ) 