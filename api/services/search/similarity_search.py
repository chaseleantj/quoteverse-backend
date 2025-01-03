from typing import List, Tuple
from sqlalchemy import select
from sqlalchemy.sql import text

from api.models.models import SessionLocal, QuoteDB
from api.services.embeddings.embeddings_processor import EmbeddingsProcessor
from api.models.models import MotivationalQuote


def similarity_search(
    query_texts: List[str], 
    k: int = 5, 
    max_distance: float = None,
    processor: EmbeddingsProcessor = None
) -> List[List[Tuple[MotivationalQuote, float]]]:
    """
    Perform similarity search using cosine distance.
    
    Args:
        query_texts (List[str]): List of query texts to search for
        k (int): Number of similar quotes to return
        max_distance (float, optional): Maximum cosine distance threshold for results
        processor (EmbeddingsProcessor, optional): Embeddings processor instance
        
    Returns:
        List[List[Tuple[MotivationalQuote, float]]]: List of lists containing similar quotes and their distances
    """
    if not processor:
        processor = EmbeddingsProcessor()
    
    query_embeddings = processor.generate_embeddings(query_texts)
    results = []
    
    with SessionLocal() as db:
        for query_vector in query_embeddings:
            distance_calc = QuoteDB.embeddings.cosine_distance(query_vector)
            stmt = select(
                QuoteDB,
                distance_calc.label('distance')
            ).where(
                distance_calc <= max_distance if max_distance is not None else True
            ).order_by('distance').limit(k)
            
            # sql = stmt.compile(
            #     dialect=db.bind.dialect,
            #     compile_kwargs={"literal_binds": True}
            # )
            # explain_results = db.execute(
            #     text(f"EXPLAIN ANALYZE {str(sql)}")
            # ).all()
            
            # print("\nVector Search Query Execution Plan:")
            # for row in explain_results:
            #     print(row[0])
            
            similar_quotes = db.execute(stmt).all()
            results.append([(quote, float(distance)) for quote, distance in similar_quotes])
    
    return results