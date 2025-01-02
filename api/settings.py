import os
import dotenv
from typing import Optional
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class Settings(BaseSettings):
    
    DB_NAME: str = os.getenv("DB_NAME")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT")
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    EMPTY_DB_CONTENTS: bool = os.getenv("EMPTY_DB_CONTENTS", False)

    ALLOWED_ORIGINS: Optional[str] = os.getenv("ALLOWED_ORIGINS", None)

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    EMBEDDING_BATCH_SIZE: int = os.getenv("EMBEDDING_BATCH_SIZE", 2048)
    EMBEDDING_DIMENSIONS: int = os.getenv("EMBEDDING_DIMENSIONS", 512)

    DATA_PATH: Optional[str] = os.getenv("DATA_PATH", None)
    PROCESSOR_PATH: Optional[str] = os.getenv("PROCESSOR_PATH", None)

    MAX_ENTRIES: int = os.getenv("MAX_ENTRIES", 5000)
    LEVENSHTEIN_SEARCH_THRESHOLD: int = os.getenv("LEVENSHTEIN_SEARCH_THRESHOLD", 3)

    LOG_TIME_IT: bool = os.getenv("LOG_TIME_IT", False)
    

settings: Settings = Settings()
