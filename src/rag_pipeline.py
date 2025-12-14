"""Main RAG pipeline that ties everything together."""
from typing import Dict, List, Optional
import logging
import os
from .document_loader import load_documents
from .embeddings import TextChunker, EmbeddingModel
from .vector_store import VectorStore
from .retrieval import Retriever
from .llm_interface import LLMInterface


logger = logging.getLogger(__name__)


class RAGPipeline:
    """The complete pipeline: load docs → chunk → embed → index → retrieve → generate answer."""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        top_k: int = 5
    ):
        """Set up all the components."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize components
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.vector_store = VectorStore()  # Auto-detect dimension
        self.retriever = None
        self.llm = None
        
        self.llm_model_name = llm_model_name
        self.is_indexed = False
    
    def index_documents(self, apple_pdf: str, tesla_pdf: str, save_path: str = None):
        """Parse PDFs, chunk them, create embeddings, and build the vector index."""
        logger.info("Indexing documents")
        
        # Load documents
        pages_data = load_documents(apple_pdf, tesla_pdf)
        
        # Chunk documents
        logger.info("Chunking documents")
        chunks = self.chunker.chunk_documents(pages_data)
        logger.info("Created %s chunks", len(chunks))
        
        # Generate embeddings
        logger.info("Generating embeddings")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        logger.info("Generated embeddings with shape %s", getattr(embeddings, "shape", None))
        
        # Add to vector store
        logger.info("Building vector store")
        self.vector_store.add_documents(chunks, embeddings)
        
        # Save if path provided
        if save_path:
            self.vector_store.save(save_path)
        
        self.is_indexed = True
        logger.info("Indexing complete")
    
    def load_index(self, load_path: str):
        """
        Load pre-built index.
        
        Args:
            load_path: Path to saved index
        """
        logger.info("Loading index from %s", load_path)
        self.vector_store.load(load_path)
        self.is_indexed = True
    
    def initialize_retriever(self):
        """Initialize retriever with re-ranker."""
        if not self.is_indexed:
            raise ValueError("Documents must be indexed before initializing retriever")
        
        logger.info("Initializing retriever")
        self.retriever = Retriever(
            self.vector_store,
            self.embedding_model,
            use_reranker=True
        )
        logger.info("Retriever ready")
    
    def initialize_llm(self):
        """Initialize LLM."""
        logger.info("Initializing LLM")
        self.llm = LLMInterface(self.llm_model_name)
        logger.info("LLM ready")
    
    def answer_question(self, query: str) -> Dict[str, any]:
        """Answer a question using retrieval + LLM generation."""
        # Make sure everything is ready
        if not self.is_indexed:
            raise ValueError("Documents must be indexed first")
        
        if self.retriever is None:
            self.initialize_retriever()
        
        if self.llm is None:
            self.initialize_llm()
        
        # Find relevant chunks
        chunks = self.retriever.retrieve(query, top_k=self.top_k)
        
        if not chunks:
            return {
                "answer": "This question cannot be answered based on the provided documents.",
                "sources": []
            }
        
        # Build context and get citation
        context = self.retriever.format_context(chunks)
        sources = self.retriever.extract_sources(chunks)
        
        # Ask LLM to generate answer
        answer = self.llm.answer_with_context(query, context)
        
        # If LLM refused, return empty sources
        if self.llm.is_refusal(answer):
            return {"answer": answer, "sources": []}
        
        return {"answer": answer, "sources": sources}


_PIPELINE: Optional[RAGPipeline] = None


def answer_question(query: str) -> dict:
    """The required assignment interface. Takes a question, returns answer + sources.
    
    Returns format: {"answer": "...", "sources": ["Apple 10-K", "Item 8", "p. 28"]}
    For out-of-scope questions, sources will be an empty list.
    """
    global _PIPELINE

    apple_pdf = os.getenv("APPLE_PDF", "data/10-Q4-2024-As-Filed.pdf")
    tesla_pdf = os.getenv("TESLA_PDF", "data/tsla-20231231-gen.pdf")
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "vector_store")

    if _PIPELINE is None:
        _PIPELINE = create_pipeline()

        if os.path.exists(vector_store_path):
            _PIPELINE.load_index(vector_store_path)
        else:
            _PIPELINE.index_documents(
                apple_pdf=apple_pdf,
                tesla_pdf=tesla_pdf,
                save_path=vector_store_path,
            )

    return _PIPELINE.answer_question(query)


def create_pipeline(
    embedding_model: str = None,
    llm_model: str = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    top_k: int | None = None
) -> RAGPipeline:
    """
    Factory function to create RAG pipeline with env defaults.
    
    Args:
        embedding_model: Override embedding model
        llm_model: Override LLM model
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        top_k: Number of documents to retrieve
        
    Returns:
        Initialized RAGPipeline
    """
    # Use env vars as defaults
    if embedding_model is None:
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    if llm_model is None:
        llm_model = (
            os.getenv("HF_MODEL")
            or os.getenv("LLM_MODEL")  # backward-compatible
            or "Qwen/Qwen2.5-1.5B-Instruct"
        )

    if chunk_size is None:
        chunk_size = int(os.getenv("CHUNK_SIZE", "800"))

    if chunk_overlap is None:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

    if top_k is None:
        top_k = int(os.getenv("TOP_K", "5"))
    
    return RAGPipeline(
        embedding_model_name=embedding_model,
        llm_model_name=llm_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k
    )
