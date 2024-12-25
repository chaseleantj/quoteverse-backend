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
    text = Column(String)
    date_created = Column(DateTime)
    embeddings = Column(ARRAY(Float))  # Store embeddings as array
    reduced_embeddings = Column(ARRAY(Float))  # Store reduced embeddings as array

# Pydantic Model
class MotivationalQuote(BaseModel):
    author: Optional[str] = None
    text: str
    date_created: datetime
    embeddings: Optional[List[float]] = None
    reduced_embeddings: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True

    def to_db_model(self) -> QuoteDB:
        return QuoteDB(
            author=self.author,
            text=self.text,
            date_created=self.date_created,
            embeddings=self.embeddings,
            reduced_embeddings=self.reduced_embeddings
        ) 