# RAG System for 10-K Document Q&A

A question-answering system for Apple's 2024 and Tesla's 2023 10-K filings. Built for the RAG + LLM assignment.


You'll need a free HuggingFace token from https://huggingface.co/settings/tokens

**Local setup:**

```bash
cd rag_project
pip install -r requirements.txt

# Add your HuggingFace token
cp .env.example .env
# Edit .env: HF_TOKEN=your_token_here

python run.py
```

This creates `outputs/answers.json` with all 13 answers.

## How It Works

1. Parse PDFs and split into chunks (800 characters each)
2. Create embeddings using sentence-transformers
3. Store in FAISS vector database
4. For each question:
   - Find top 15 similar chunks
   - Re-rank to get best 5
   - Send to LLM with context
   - Get answer with citations

**Models used:**
- Embeddings: all-MiniLM-L6-v2
- Re-ranker: ms-marco-MiniLM-L-6-v2
- LLM: Qwen 1.5B via HuggingFace API 

## The Required Function

The assignment asks for `answer_question()` that returns answers with sources:

```python
from src.rag_pipeline import answer_question

result = answer_question("What was Apple's revenue in 2024?")
# {'answer': '$391,036 million', 'sources': ['Apple 10-K', 'Item 8', 'p. 28']}
```

For questions it can't answer (like stock predictions), it returns empty sources.

## Files

- `src/` - Main code (document loading, embeddings, retrieval, LLM interface)
- `run.py` - Runs all 13 evaluation questions
- `rag_colab.ipynb` - Notebook version for Colab/Kaggle
- `requirements.txt` - Python packages needed

## What It Does (Assignment Requirements)

- Parses both PDFs keeping track of doc name, section, and page number
- Uses open-source embeddings (no paid APIs)
- Stores in FAISS vector database
- Retrieves top 5 chunks using two-stage search
- Uses open-access LLM (not GPT-4 or Claude)
- Cites sources like ["Apple 10-K", "Item 8", "p. 28"]
- Refuses to answer out-of-scope questions
- Works in Colab/Kaggle notebooks
