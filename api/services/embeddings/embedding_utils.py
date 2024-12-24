import pandas as pd
from typing import List
import numpy as np

def validate_embeddings(embeddings: List[List[float]]) -> bool:
    """
    Validate that embeddings are in the correct format and dimensions match.
    
    Args:
        embeddings (List[List[float]]): List of embedding vectors
        
    Returns:
        bool: True if embeddings are valid
        
    Raises:
        ValueError: If embeddings are invalid
    """
    if not embeddings:
        raise ValueError("Embeddings list is empty")
        
    dim = len(embeddings[0])
    if not all(len(emb) == dim for emb in embeddings):
        raise ValueError("All embedding vectors must have the same dimension")
        
    return True

def calculate_embedding_statistics(embeddings: List[List[float]]) -> dict:
    """
    Calculate basic statistics about the embeddings.
    
    Args:
        embeddings (List[List[float]]): List of embedding vectors
        
    Returns:
        dict: Dictionary containing embedding statistics
    """
    embeddings_array = np.array(embeddings)
    return {
        "mean": np.mean(embeddings_array),
        "std": np.std(embeddings_array),
        "min": np.min(embeddings_array),
        "max": np.max(embeddings_array),
        "dimension": embeddings_array.shape[1]
    } 