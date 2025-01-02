from typing import List, Literal
from sqlalchemy import select, func, cast, String
from sqlalchemy.dialects.postgresql import TEXT as PGText
from sqlalchemy.orm import Session

from api.models.models import SessionLocal, QuoteDB
from api.settings import settings


def transform_text(text: str) -> str:
    return text.lower().replace(' ', '')


def transform_text_sql(column):
    return func.lower(func.replace(column, ' ', ''))


def text_search(
    search_texts: List[str], 
    search_by: Literal["author", "book"],
    k: int = 5, 
    strict: bool = False
) -> List[List[QuoteDB]]:
    """
    Returns quotes by searching either the author or book field using pg_trgm
    fuzzy matching if strict=False, or exact matching if strict=True.

    Args:
        search_texts (List[str]): Texts to search for
        search_by (str): Field to search in ("author" or "book")
        k (int): Number of quotes to return
        strict (bool): Whether to use strict comparison or fuzzy comparison

    Returns:
        List[List[QuoteDB]]: List of quotes for each search text
    """
    results = []
    # Preprocess the input strings (lowercase and remove spaces)
    transformed_texts = [transform_text(text) for text in search_texts]

    with SessionLocal() as db:
        for transformed_text in transformed_texts:
            # Build the base query for QuoteDB
            query = select(QuoteDB)
            column = getattr(QuoteDB, search_by)

            if strict:
                # Strict equality on the transformed text
                # (i.e., exact match on lowercased text without spaces)
                query = query.where(transform_text_sql(column) == transformed_text)
            else:
                # Fuzzy matching using the % operator with explicit cast
                query = (
                    query
                    .where(
                        cast(transform_text_sql(column), PGText).op('%')(
                            cast(transformed_text, PGText)
                        )
                    )
                    .order_by(
                        func.similarity(
                            cast(transform_text_sql(column), PGText),
                            cast(transformed_text, PGText)
                        ).desc()
                    )
                )

            # Limit the query and fetch results
            quotes = db.execute(query.limit(k)).scalars().all()

            # (Optional) Fallback to partial contains if no results found
            if not quotes and len(transformed_text) > 5:
                contains_query = (
                    select(QuoteDB)
                    .where(
                        transform_text_sql(column).contains(transformed_text)
                    )
                    .limit(k)
                )
                quotes = db.execute(contains_query).scalars().all()

            results.append(quotes)

    return results
