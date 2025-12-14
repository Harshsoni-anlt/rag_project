"""Text chunking and embeddings using sentence-transformers."""

from typing import Dict, List

import logging

import os

import numpy as np


logger = logging.getLogger(__name__)


class TextChunker:
    """Splits text into chunks with overlap so we don't lose context at boundaries."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        """chunk_size: max chars per chunk, chunk_overlap: how much to repeat between chunks"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_documents(self, pages_data: List[Dict]) -> List[Dict]:
        """Break pages into chunks, keeping the metadata (doc name, section, page)."""
        chunks = []
        
        for page in pages_data:
            text = page['text']
            page_num = page['page']
            doc_name = page['doc_name']
            section = page['section']
            
            # Split text into chunks
            page_chunks = self._split_text(text)
            
            for chunk_text in page_chunks:
                chunks.append({
                    'text': chunk_text,
                    'page': page_num,
                    'doc_name': doc_name,
                    'section': section
                })
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split long text into overlapping chunks, trying to break at sentence boundaries."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at a period or newline instead of mid-sentence
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return chunks


class EmbeddingModel:
    """Converts text to vectors using sentence-transformers (runs locally, no API)."""
    
    def __init__(self, model_name: str = None):
        """Load the embedding model. Downloads ~80MB on first run."""
        from sentence_transformers import SentenceTransformer

        model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding model ready (%s dimensions)", self.embedding_dim)
    
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Turn a list of texts into embedding vectors."""
        return self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            batch_size=32
        )
