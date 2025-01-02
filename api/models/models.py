from datetime import datetime
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel

from api.settings import settings

# SQLAlchemy setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Model
class QuoteDB(Base):
    __tablename__ = "motivational_quotes"

    id = Column(Integer, primary_key=True, index=True)
    author = Column(String)
    book = Column(String)
    text = Column(String)
    date_created = Column(DateTime)
    embeddings = Column(HALFVEC(512))
    reduced_embeddings = Column(HALFVEC(2))

# Pydantic Model
class MotivationalQuote(BaseModel):
    author: Optional[str] = None
    book: Optional[str] = None
    text: str
    date_created: datetime
    embeddings: Optional[List[float]] = None
    reduced_embeddings: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True

    def to_db_model(self) -> QuoteDB:
        return QuoteDB(
            author=self.author,
            book=self.book,
            text=self.text,
            date_created=self.date_created,
            embeddings=self.embeddings,
            reduced_embeddings=self.reduced_embeddings
        )
