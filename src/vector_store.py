"""Vector store using FAISS for fast similarity search."""
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple
import logging
import os


logger = logging.getLogger(__name__)


class VectorStore:
    """Stores embeddings and searches for similar ones using FAISS."""
    
    def __init__(self, embedding_dim: int = None):
        """embedding_dim: size of vectors (auto-detected from first batch if not set)"""
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []
    
    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        """Add chunks and their embeddings to the store."""
        # Create the FAISS index if this is the first batch
        if self.index is None:
            self.embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Normalize so we can use L2 distance as cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype('float32'))
        self.chunks = chunks

        logger.info("Added %s chunks to vector store", len(chunks))
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Find the most similar chunks to the query."""
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Turn distances into similarity scores
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                similarity = 1 - (dist / 2)  # L2 distance -> similarity score
                results.append((self.chunks[idx], float(similarity)))
        
        return results
    
    def save(self, path: str):
        """Save to disk so we don't have to rebuild the index every time."""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # Save chunks metadata
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        
        logger.info("Vector store saved to %s", path)
    
    def load(self, path: str):
        """Load index and chunks from disk."""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        # Load chunks metadata
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        
        logger.info("Vector store loaded from %s", path)
    
    @property
    def size(self) -> int:
        """Get number of documents in store."""
        return len(self.chunks)
