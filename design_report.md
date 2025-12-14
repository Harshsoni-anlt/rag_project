# Design Report

## Chunking Strategy

I went with 800 characters per chunk with 200 characters of overlap (25%).

Why 800? Financial docs have tables and long explanations. 800 chars is about 5-6 sentences, which usually contains a complete idea or table section. If you go smaller (like 500), you break up context too much. If you go bigger (1200), the chunks become less precise for retrieval.

The 25% overlap means if an important number is near a boundary, it'll show up in two chunks. This helps avoid missing things due to unlucky splits.

## Embedding Model

I'm using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).

It's only about 80MB, runs fast on CPU, and works well for finding similar text. Since we only have two documents to search (~2000 chunks total), 384 dimensions is plenty. I tried the bigger `all-mpnet-base-v2` model (768-dim) but it was twice as slow and didn't really help accuracy for this specific task.

## Retrieval (Two Stages)

First stage: FAISS finds the top 15 most similar chunks using vector search.

Second stage: A cross-encoder model re-scores those 15 and picks the best 5.

Why bother with two stages? The embedding model (bi-encoder) is fast but not super precise - it creates embeddings separately for the question and chunks. The cross-encoder actually looks at question+chunk together and scores how well they match. This is slower but way more accurate. In my testing, adding the re-ranker improved accuracy by about 20%, especially for questions asking about percentages or specific details.

Why 15→5? The re-ranker needs enough candidates to find the good ones, but we don't want to send too much context to the LLM.

## LLM Choice

I'm using `Qwen/Qwen2.5-1.5B-Instruct` through HuggingFace's Inference API.

The assignment said "local or open-access" LLM. I went with open-access (HF's free API) because:
- Works in Colab/Kaggle without needing a GPU
- No 7GB+ model download
- Qwen 1.5B is small but good at following instructions
- It's open-source (Apache 2.0)

I tried Mistral-7B too but it was 3-4x slower on the free tier and didn't really give better answers for this task.

## Handling Out-of-Scope Questions

The prompt tells the LLM:
- If the answer isn't in the documents: say "Not specified in the document."
- If the question is unanswerable (predictions, current info, irrelevant stuff): say "This question cannot be answered based on the provided documents."

The test questions check this:
- Q11 asks for stock forecast (future prediction) → should refuse
- Q12 asks for current CFO "as of 2025" but the filing is from 2024 → should refuse  
- Q13 asks what color the headquarters is (irrelevant) → should refuse

When the LLM refuses, the code detects those phrases and returns an empty sources list.

## Source Citations

Format: `["Apple 10-K", "Item 8", "p. 28"]` as three separate strings.

When parsing PDFs, I save the document name, section, and page number for each chunk. After retrieval, the system uses the metadata from the top-ranked chunk as the citation. This way the grader can look up the answer in the actual PDF.
