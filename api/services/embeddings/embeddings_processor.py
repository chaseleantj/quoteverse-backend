import os
from typing import List, Optional, Literal, Union, Tuple

import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
from openai import OpenAI

from api.models.models import MotivationalQuote
from api.services.utils.utils import time_it


class EmbeddingsProcessor:
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "text-embedding-3-small",
                 reduction_method: Literal["umap", "pca"] = "umap"):
        """
        Initialize the EmbeddingsProcessor.
        
        Args:
            api_key (str): OpenAI API key
            model (str): OpenAI embedding model to use
            reduction_method (str): Dimension reduction method ("umap" or "pca")
        """
        self._initialize_client(api_key)
        self.model = model
        self.reduction_method = reduction_method
        self.dim_reducer: Optional[Union[PCA, umap.UMAP]] = None
        self.standardization_params = {
            'mean': None,
            'std': None
        }
        
    def _initialize_client(self, api_key: Optional[str] = None) -> None:
        """Initialize OpenAI client - separated to handle serialization"""
        self.client = OpenAI(api_key=api_key if api_key else os.getenv("OPENAI_API_KEY"))
    
    def __getstate__(self):
        """Custom serialization method"""
        state = self.__dict__.copy()
        # Don't pickle the OpenAI client
        del state['client']
        return state
    
    def __setstate__(self, state):
        """Custom deserialization method"""
        self.__dict__.update(state)
        # Reinitialize the OpenAI client
        self._initialize_client()
    
    @time_it
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI API.
        
        Args:
            texts (List[str]): List of texts to generate embeddings for
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        embeddings = []
        
        # Process texts in batches to avoid API limits
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", 100))
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
        return embeddings
    
    @time_it
    def fit_reducer(self, 
                   embeddings: List[List[float]], 
                   n_components: int) -> None:
        """
        Fit dimension reduction transformer and compute standardization parameters.
        """
        # Convert embeddings to numpy array with explicit dtype
        embeddings_array = np.array(embeddings, dtype=np.float64)
        
        if self.reduction_method == "pca":
            self.dim_reducer = PCA(n_components=n_components)
        else:  # umap
            self.dim_reducer = umap.UMAP(
                n_neighbors=min(int(np.sqrt(len(embeddings))), len(embeddings) - 1),
                n_components=n_components,
                metric='cosine'
            )
            
        # Fit the reducer
        reduced_data = self.dim_reducer.fit_transform(embeddings_array)
        
        # Calculate standardization parameters
        self.standardization_params['mean'] = np.mean(reduced_data, axis=0)
        self.standardization_params['std'] = np.std(reduced_data, axis=0)

    @time_it
    def transform_embeddings(self, 
                           embeddings: List[List[float]]) -> np.ndarray:
        """
        Transform embeddings using fitted dimension reducer and standardize.
        """
        if self.dim_reducer is None:
            raise ValueError("Dimension reducer must be fitted before transformation")
            
        # Transform the embeddings
        reduced_data = self.dim_reducer.transform(embeddings)
        
        # Standardize the output
        standardized_data = (reduced_data - self.standardization_params['mean']) / self.standardization_params['std']
        
        return standardized_data
    
    def add_embeddings_to_quotes(self, quotes: List[MotivationalQuote]) -> List[MotivationalQuote]:
        """
        Generate embeddings for a list of motivational quotes.
        
        Args:
            quotes (List[MotivationalQuote]): List of quotes to process
            
        Returns:
            List[MotivationalQuote]: Quotes with embeddings added
        """
        # Extract quotes text
        texts = [quote.text for quote in quotes]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Update quotes with embeddings
        processed_quotes = []
        for quote, embedding in zip(quotes, embeddings):
            quote_copy = quote.model_copy()
            quote_copy.embeddings = embedding
            processed_quotes.append(quote_copy)
            
        return processed_quotes
    
    def add_reduced_embeddings_to_quotes(self, 
                         quotes: List[MotivationalQuote], 
                         n_components: int = 2) -> List[MotivationalQuote]:
        """
        Add dimension-reduced embeddings to quotes.
        
        Args:
            quotes (List[MotivationalQuote]): List of quotes with embeddings
            n_components (int): Number of components to reduce to
            
        Returns:
            List[MotivationalQuote]: Quotes with reduced embeddings added
        """
        # Extract embeddings
        embeddings = [quote.embeddings for quote in quotes if quote.embeddings is not None]
        
        if not embeddings:
            raise ValueError("No embeddings found in quotes")
        
        # Fit reducer if not already fitted
        if self.dim_reducer is None:
            self.fit_reducer(embeddings, n_components)
        
        # Transform embeddings
        reduced_embeddings = self.transform_embeddings(embeddings)
        
        # Update quotes with reduced embeddings
        processed_quotes = []
        for quote, reduced_embedding in zip(quotes, reduced_embeddings):
            quote_copy = quote.model_copy()
            quote_copy.reduced_embeddings = reduced_embedding.tolist()
            processed_quotes.append(quote_copy)
            
        return processed_quotes 