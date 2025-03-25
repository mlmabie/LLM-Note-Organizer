"""
Embedding service for text processing.

This module provides functions for generating and using text embeddings,
including support for Clustered Compositional Embeddings (CCE) for more
efficient vector storage and processing.
"""

import hashlib
import pickle
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
import cce

from note_organizer.core.config import settings
from note_organizer.db.database import get_session, get_from_cache, set_in_cache
from note_organizer.db.models import EmbeddingCache


class EmbeddingService:
    """Service for generating and using text embeddings."""

    def __init__(self, model_name: Optional[str] = None, use_cce: Optional[bool] = None):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            use_cce: Whether to use Clustered Compositional Embeddings
        """
        self.model_name = model_name or settings.embedding.model_name
        self.use_cce = use_cce if use_cce is not None else settings.embedding.use_cce
        
        # Initialize the SentenceTransformer model (lazy loaded)
        self._model = None
        
        # Initialize CCE clusterer (lazy loaded)
        self._cce_clusterer = None
        self.cce_centroids = settings.embedding.cce_centroids
        self.cce_dim = settings.embedding.cce_dim
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the SentenceTransformer model."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    @property
    def cce_clusterer(self):
        """Get the CCE clusterer."""
        if not self.use_cce:
            return None
            
        if self._cce_clusterer is None:
            logger.info(f"Initializing CCE clusterer with {self.cce_centroids} centroids")
            self._cce_clusterer = cce.CCE(
                self.model.get_sentence_embedding_dimension(),
                d=self.cce_dim,
                num_clusters=self.cce_centroids,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        return self._cce_clusterer
    
    def get_embedding(
        self, text: str, use_cache: bool = True, compress: bool = True
    ) -> np.ndarray:
        """Get embedding for a text.
        
        Args:
            text: The text to embed
            use_cache: Whether to use cache
            compress: Whether to compress the embedding with CCE
            
        Returns:
            The embedding as a numpy array
        """
        # Create a hash of the text for caching
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"embedding:{self.model_name}:{compress}:{text_hash}"
        
        # Check in-memory cache first
        if use_cache:
            cached = get_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Check database cache
        if use_cache:
            with get_session() as session:
                db_cache = session.query(EmbeddingCache).filter(
                    EmbeddingCache.text_hash == text_hash,
                    EmbeddingCache.model_name == f"{self.model_name}{'_cce' if compress else ''}"
                ).first()
                
                if db_cache:
                    embedding = pickle.loads(db_cache.embedding)
                    # Store in memory cache for faster access next time
                    set_in_cache(cache_key, embedding)
                    return embedding
        
        # Generate embedding
        embedding = self.model.encode([text])[0]
        
        # Compress using CCE if requested
        if compress and self.use_cce:
            embedding = self.compress_embedding(embedding)
        
        # Store in database cache
        if use_cache:
            with get_session() as session:
                # Check if it already exists (could have been added in another process)
                existing = session.query(EmbeddingCache).filter(
                    EmbeddingCache.text_hash == text_hash,
                    EmbeddingCache.model_name == f"{self.model_name}{'_cce' if compress else ''}"
                ).first()
                
                if not existing:
                    db_cache = EmbeddingCache(
                        text_hash=text_hash,
                        embedding=pickle.dumps(embedding),
                        model_name=f"{self.model_name}{'_cce' if compress else ''}"
                    )
                    session.add(db_cache)
                    session.commit()
        
        # Store in memory cache
        set_in_cache(cache_key, embedding)
        
        return embedding
    
    def get_embeddings(
        self, texts: List[str], use_cache: bool = True, compress: bool = True
    ) -> np.ndarray:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache
            compress: Whether to compress embeddings with CCE
            
        Returns:
            Array of embeddings
        """
        # This could be optimized further by batching uncached texts
        return np.array([self.get_embedding(text, use_cache, compress) for text in texts])
    
    def compress_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Compress an embedding using CCE.
        
        Args:
            embedding: The embedding to compress
            
        Returns:
            Compressed embedding
        """
        if not self.use_cce:
            return embedding
        
        # Convert to tensor if it's a numpy array
        if isinstance(embedding, np.ndarray):
            embedding_tensor = torch.from_numpy(embedding).float()
        else:
            embedding_tensor = embedding
            
        # Add batch dimension if needed
        if embedding_tensor.dim() == 1:
            embedding_tensor = embedding_tensor.unsqueeze(0)
            
        # Compress
        compressed = self.cce_clusterer.encode(embedding_tensor)
        
        # Convert back to numpy and remove batch dimension if added
        compressed_np = compressed.cpu().numpy()
        if compressed_np.shape[0] == 1:
            compressed_np = compressed_np[0]
            
        return compressed_np
    
    def decompress_embedding(self, compressed: np.ndarray) -> np.ndarray:
        """Decompress a CCE embedding.
        
        Args:
            compressed: The compressed embedding
            
        Returns:
            Decompressed embedding
        """
        if not self.use_cce:
            return compressed
        
        # Convert to tensor if it's a numpy array
        if isinstance(compressed, np.ndarray):
            compressed_tensor = torch.from_numpy(compressed).float()
        else:
            compressed_tensor = compressed
            
        # Add batch dimension if needed
        if compressed_tensor.dim() == 1:
            compressed_tensor = compressed_tensor.unsqueeze(0)
            
        # Decompress
        decompressed = self.cce_clusterer.decode(compressed_tensor)
        
        # Convert back to numpy and remove batch dimension if added
        decompressed_np = decompressed.cpu().numpy()
        if decompressed_np.shape[0] == 1:
            decompressed_np = decompressed_np[0]
            
        return decompressed_np
    
    def similarity(
        self, 
        text1: Union[str, np.ndarray], 
        text2: Union[str, np.ndarray],
        compressed: bool = False
    ) -> float:
        """Calculate similarity between two texts or embeddings.
        
        Args:
            text1: First text or embedding
            text2: Second text or embedding
            compressed: Whether the embeddings are already compressed
            
        Returns:
            Similarity score (0-1)
        """
        # Get embeddings if texts were provided
        if isinstance(text1, str):
            embedding1 = self.get_embedding(text1, compress=compressed)
        else:
            embedding1 = text1
            
        if isinstance(text2, str):
            embedding2 = self.get_embedding(text2, compress=compressed)
        else:
            embedding2 = text2
        
        # If compressed and we need to decompress for proper similarity
        if compressed and self.use_cce and not self.cce_clusterer.metric_in_z_space:
            embedding1 = self.decompress_embedding(embedding1)
            embedding2 = self.decompress_embedding(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def find_most_similar(
        self, 
        query: Union[str, np.ndarray], 
        candidates: List[Union[str, np.ndarray]],
        compressed: bool = False,
        top_k: int = 1
    ) -> List[Tuple[int, float]]:
        """Find the most similar items to a query.
        
        Args:
            query: Query text or embedding
            candidates: List of candidate texts or embeddings
            compressed: Whether the embeddings are already compressed
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity) pairs, ordered by similarity
        """
        # Get query embedding if text was provided
        if isinstance(query, str):
            query_embedding = self.get_embedding(query, compress=compressed)
        else:
            query_embedding = query
        
        # Get candidate embeddings
        candidate_embeddings = []
        for candidate in candidates:
            if isinstance(candidate, str):
                candidate_embedding = self.get_embedding(candidate, compress=compressed)
            else:
                candidate_embedding = candidate
            candidate_embeddings.append(candidate_embedding)
        
        # Calculate similarities
        similarities = [
            self.similarity(query_embedding, candidate_embedding, compressed=compressed)
            for candidate_embedding in candidate_embeddings
        ]
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(idx, similarities[idx]) for idx in top_indices]


# Singleton instance for reuse
embedding_service = EmbeddingService() 