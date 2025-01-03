from typing import List, Literal
from sqlalchemy import select, func, cast, text, String
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
    # No longer need to transform input texts
    with SessionLocal() as db:
        for search_text in search_texts:  # Using original search_text directly
            query = select(QuoteDB)
            column = getattr(QuoteDB, search_by)

            if strict:
                # Direct equality comparison without transformation
                query = query.where(column == search_text)
            else:
                # Fuzzy matching using original text
                query = (
                    query
                    .where(
                        cast(column, PGText).op('%')(
                            cast(search_text, PGText)
                        )
                    )
                    .order_by(
                        func.similarity(
                            cast(column, PGText),
                            cast(search_text, PGText)
                        ).desc()
                    )
                )

            # # Print the EXPLAIN ANALYZE plan
            # explain_query = text(f"EXPLAIN ANALYZE {query.compile(compile_kwargs={'literal_binds': True})}")
            # result = db.execute(explain_query)
            # for row in result:
            #     print(row[0])
            
            # # Limit the query and fetch results
            quotes = db.execute(query.limit(k)).scalars().all()

            # (Optional) Fallback to partial contains if no results found
            if not quotes and len(search_text) > 5:
                contains_query = (
                    select(QuoteDB)
                    .where(column.contains(search_text))
                    .limit(k)
                )
                quotes = db.execute(contains_query).scalars().all()

            results.append(quotes)

    return results
