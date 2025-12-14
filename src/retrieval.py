"""Two-stage retrieval: vector search then re-ranking."""
from typing import List, Dict, Tuple
import logging
import numpy as np
from sentence_transformers import CrossEncoder


logger = logging.getLogger(__name__)


class Retriever:
    """Finds relevant chunks using vector search + cross-encoder re-ranking."""
    
    def __init__(self, vector_store, embedding_model, use_reranker: bool = True):
        """Set up retriever. use_reranker=True for better accuracy (recommended)."""
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.use_reranker = use_reranker
        
        if use_reranker:
            logger.info("Loading cross-encoder re-ranker")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Re-ranker ready")
        else:
            self.reranker = None
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Get the most relevant chunks for a question."""
        # First get more candidates than we need (for re-ranking)
        search_k = top_k * 3 if self.use_reranker else top_k
        query_embedding = self.embedding_model.encode([query], show_progress=False)
        
        initial_results = self.vector_store.search(query_embedding, top_k=search_k)
        
        # Re-rank to get the actual best ones
        if self.use_reranker and len(initial_results) > 0:
            results = self._rerank(query, initial_results, top_k)
        else:
            results = [chunk for chunk, score in initial_results[:top_k]]
        
        return results
    
    def _rerank(self, query: str, candidates: List[Tuple[Dict, float]], top_k: int) -> List[Dict]:
        """Use cross-encoder to score query+chunk pairs more accurately."""
        pairs = [[query, chunk['text']] for chunk, score in candidates]
        
        ce_scores = self.reranker.predict(pairs)
        
        # Sort by cross-encoder score
        reranked = sorted(
            zip([chunk for chunk, _ in candidates], ce_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [chunk for chunk, score in reranked[:top_k]]
    
    def format_context(self, chunks: List[Dict]) -> str:
        """Turn chunks into a formatted string to send to the LLM."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = f"[{chunk['doc_name']}, {chunk['section']}, p. {chunk['page']}]"
            context_parts.append(f"Source {i} {source}:\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
    
    def extract_sources(self, chunks: List[Dict]) -> List[str]:
        """Get source citation from the top chunk in the format ["Apple 10-K", "Item 8", "p. 28"]"""
        if not chunks:
            return []

        chunk = chunks[0]
        return [
            str(chunk.get("doc_name", "")),
            str(chunk.get("section", "")),
            f"p. {chunk.get('page', '')}",
        ]
