from typing import List
from sqlalchemy import select, func

from api.models.models import SessionLocal, QuoteDB
from api.settings import settings


def transform_author(author: str) -> str:
    return author.lower().replace(' ', '')


def transform_author_sql(column):
    return func.lower(func.replace(column, ' ', ''))


def author_search(authors: List[str], k: int = 5, strict: bool = False) -> List[List[QuoteDB]]:
    """
    Returns quotes by the specified author.
    
    Args:
        author (str): Author to search for
        k (int): Number of quotes to return
        strict (bool): Whether to use strict comparison, which compares the author name exactly, or fuzzy comparison, which allows for slight variations in spelling and length.
        
    Returns:
        List[QuoteDB]: List of quotes
    """
    results = []
    transformed_authors = [transform_author(author) for author in authors]

    with SessionLocal() as db:
        for transformed_author in transformed_authors:
            query = select(QuoteDB)

            if strict:
                query = query.where(transform_author_sql(QuoteDB.author) == transformed_author)
            else:
                query = query.where(
                    func.levenshtein(
                        transform_author_sql(QuoteDB.author), transformed_author
                    ) <= settings.LEVENSHTEIN_SEARCH_THRESHOLD
                )

            quotes = db.execute(query.limit(k)).scalars().all()
            results.append(quotes)

    return results
