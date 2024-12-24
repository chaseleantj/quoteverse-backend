import os
import pandas as pd
import numpy as np
from openai import OpenAI
from typing import List, Optional
from sklearn.decomposition import PCA
from api.models.models import MotivationalQuote


class EmbeddingsProcessor:
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize the EmbeddingsProcessor.
        
        Args:
            api_key (str): OpenAI API key
            model (str): OpenAI embedding model to use
        """
        self.client = OpenAI(api_key=api_key if api_key else os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.pca_transformer: Optional[PCA] = None
        
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
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
        return embeddings
    
    def add_embeddings_to_df(self, 
                            df: pd.DataFrame, 
                            text_column: str, 
                            embedding_column: str = 'embeddings') -> pd.DataFrame:
        """
        Add embeddings as a new column to the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of the column containing text to embed
            embedding_column (str): Name of the new column to store embeddings
            
        Returns:
            pd.DataFrame: DataFrame with embeddings column added
        """
        texts = df[text_column].tolist()
        embeddings = self.generate_embeddings(texts)
        df_copy = df.copy()
        df_copy[embedding_column] = embeddings
        return df_copy
    
    def fit_pca(self, 
                embeddings: List[List[float]], 
                n_components: int) -> None:
        """
        Fit PCA transformer on embedding vectors.
        
        Args:
            embeddings (List[List[float]]): List of embedding vectors
            n_components (int): Number of components to reduce to
        """
        self.pca_transformer = PCA(n_components=n_components)
        self.pca_transformer.fit(embeddings)
        
    def transform_embeddings(self, 
                           embeddings: List[List[float]]) -> np.ndarray:
        """
        Transform embeddings using fitted PCA transformer.
        
        Args:
            embeddings (List[List[float]]): List of embedding vectors
            
        Returns:
            np.ndarray: Reduced dimensionality embeddings
        """
        if self.pca_transformer is None:
            raise ValueError("PCA transformer must be fitted before transformation")
        return self.pca_transformer.transform(embeddings)
    
    def add_pca_embeddings_to_df(self,
                                df: pd.DataFrame,
                                embeddings_column: str,
                                n_components: int,
                                pca_column: str = 'pca_embeddings') -> pd.DataFrame:
        """
        Add PCA-reduced embeddings as a list in a new column to the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            embeddings_column (str): Name of the column containing embeddings
            n_components (int): Number of components to reduce to
            pca_column (str): Name of the new column to store PCA embeddings
            
        Returns:
            pd.DataFrame: DataFrame with PCA embedding list column added
        """
        embeddings = df[embeddings_column].tolist()
        
        # Fit PCA if not already fitted
        if self.pca_transformer is None:
            self.fit_pca(embeddings, n_components)
            
        # Transform embeddings
        reduced_embeddings = self.transform_embeddings(embeddings)
        
        # Add new column to dataframe with PCA embeddings as lists
        df_copy = df.copy()
        df_copy[pca_column] = reduced_embeddings.tolist()  # Store as list
        
        return df_copy 
    
    def get_pca_coordinates(self, text: str, n_components: int = 2) -> np.ndarray:
        """
        Get PCA coordinates for a single text string.
        
        Args:
            text (str): Text to process
            n_components (int): Number of PCA components to use
            
        Returns:
            np.ndarray: Array of PCA coordinates
            
        Raises:
            ValueError: If PCA transformer is not fitted
        """
        # Generate embedding
        embedding = self.generate_embeddings([text])
        
        # Fit PCA if not already fitted
        if self.pca_transformer is None or self.pca_transformer.n_components_ != n_components:
            raise ValueError("PCA transformer must be fitted with the correct number of components before use")
        
        # Transform to PCA coordinates
        pca_coords = self.transform_embeddings(embedding)
        return pca_coords[0]  # Return coordinates for single text 
    
    def process_quotes(self, quotes: List[MotivationalQuote]) -> List[MotivationalQuote]:
        """
        Generate embeddings for a list of motivational quotes.
        
        Args:
            quotes (List[MotivationalQuote]): List of quotes to process
            
        Returns:
            List[MotivationalQuote]: Quotes with embeddings added
        """
        # Extract quotes text
        texts = [quote.quote for quote in quotes]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Update quotes with embeddings
        processed_quotes = []
        for quote, embedding in zip(quotes, embeddings):
            quote_copy = quote.model_copy()
            quote_copy.embeddings = embedding
            processed_quotes.append(quote_copy)
            
        return processed_quotes
    
    def add_pca_to_quotes(self, 
                         quotes: List[MotivationalQuote], 
                         n_components: int = 2) -> List[MotivationalQuote]:
        """
        Add PCA-reduced embeddings to quotes.
        
        Args:
            quotes (List[MotivationalQuote]): List of quotes with embeddings
            n_components (int): Number of PCA components
            
        Returns:
            List[MotivationalQuote]: Quotes with PCA embeddings added
        """
        # Extract embeddings
        embeddings = [quote.embeddings for quote in quotes if quote.embeddings is not None]
        
        if not embeddings:
            raise ValueError("No embeddings found in quotes")
        
        # Fit PCA if not already fitted
        if self.pca_transformer is None:
            self.fit_pca(embeddings, n_components)
        
        # Transform embeddings
        reduced_embeddings = self.transform_embeddings(embeddings)
        
        # Update quotes with PCA embeddings
        processed_quotes = []
        for quote, pca_embedding in zip(quotes, reduced_embeddings):
            quote_copy = quote.model_copy()
            quote_copy.pca_embeddings = pca_embedding.tolist()
            processed_quotes.append(quote_copy)
            
        return processed_quotes 