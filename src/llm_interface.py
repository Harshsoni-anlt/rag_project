"""LLM interface using HuggingFace's free Inference API."""
import os
import re
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class LLMInterface:
    """Wraps HuggingFace Inference API for question answering."""
    
    def __init__(self, model_name: str = None):
        """Set up connection to HuggingFace API."""
        self.model_name = model_name or os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        self.hf_token = os.getenv("HF_TOKEN")
        self.hf_api_key = os.getenv("HF_API_KEY")  # alternative name for token
        # Provider routing: auto lets HF pick, legacy uses classic endpoint
        self.hf_provider = os.getenv("HF_PROVIDER", "auto")
        self.hf_base_url = os.getenv("HF_BASE_URL")
        
        msg = f"Initializing HuggingFace Inference API (provider: {self.hf_provider}, model: {self.model_name})..."
        if self.hf_base_url:
            msg = msg[:-4] + f", base_url: {self.hf_base_url})..."
        logger.info(msg)
        
        from huggingface_hub import InferenceClient

        self.client = self._make_client(provider=self.hf_provider, base_url=self.hf_base_url)

        logger.info("LLM client ready")

    def _make_client(self, provider: str | None, base_url: str | None):
        """Create HF InferenceClient with the right settings."""
        from huggingface_hub import InferenceClient

        if provider == "legacy":
            provider = None
            base_url = base_url or "https://api-inference.huggingface.co"

        kwargs = {}
        # Use whichever token name is set
        if self.hf_api_key:
            kwargs["api_key"] = self.hf_api_key
        elif self.hf_token:
            kwargs["token"] = self.hf_token
        if provider:
            kwargs["provider"] = provider
        if base_url:
            kwargs["base_url"] = base_url
        return InferenceClient(**kwargs)
    
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """Send prompt to LLM and get response."""
        # Try chat-style first (most models), then text generation as fallback
        primary_client = self.client
        fallback_clients = []

        # If the first provider fails, try alternatives
        if self.hf_provider == "hf-inference":
            fallback_clients.append(self._make_client(provider="auto", base_url=None))

        fallback_clients.append(self._make_client(provider="legacy", base_url=self.hf_base_url))

        last_chat_error = None
        last_text_error = None

        for client in [primary_client] + fallback_clients:
            try:
                content = self._generate_chat_with_client(
                    client,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                return (content or "").strip()
            except Exception as chat_error:
                last_chat_error = chat_error
                try:
                    response = client.text_generation(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        return_full_text=False,
                    )
                    return response.strip()
                except Exception as text_error:
                    last_text_error = text_error
                    continue

        msg = f"{last_text_error or last_chat_error}"
        logger.warning("LLM chat error: %s", last_chat_error)
        logger.warning("LLM text-generation error: %s", last_text_error)

        if "403" in msg or "403 Forbidden" in msg or "Inference Providers" in msg:
            return (
                "Error generating response. HuggingFace rejected the request (403). "
                "Make sure your HF token has inference permissions. If using HF_PROVIDER=auto, try HF_PROVIDER=legacy."
            )

        if "404" in msg and ("hf-inference" in msg or "api-inference.huggingface.co" in msg or "router.huggingface.co" in msg):
            return (
                "Error generating response. The selected model/provider endpoint returned 404 (not found). "
                "Fix: set HF_PROVIDER=auto or HF_PROVIDER=legacy, and/or switch HF_MODEL to a served model."
            )

        return "Error generating response."

    def _generate_chat_with_client(self, client, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Use chat API style (most models work better with this)."""
        messages = [{"role": "user", "content": prompt}]

        # Try newer API first
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            return getattr(resp.choices[0].message, "content", "") or ""

        # Fallback to older chat API
        resp = client.chat_completion(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        msg = resp.choices[0].message
        if isinstance(msg, dict):
            return (msg.get("content") or "")
        return getattr(msg, "content", "") or ""

    def _generate_chat(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        return self._generate_chat_with_client(self.client, prompt, max_new_tokens, temperature)
    
    def create_prompt(self, query: str, context: str) -> str:
        """Build the prompt we send to the LLM."""
        prompt_template = """You are a financial analyst assistant. Answer the question using ONLY the provided context from Apple's 2024 10-K and Tesla's 2023 10-K documents.

RULES:
1. Answer ONLY using information from the context
2. Cite sources EXACTLY as: ["Apple 10-K", "Item 8", "p. 28"]
3. If answer is not in context: "Not specified in the document."
4. If out of scope (predictions, current info not in filing): "This question cannot be answered based on the provided documents."
5. Be precise and concise

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt_template.format(context=context, query=query)
    
    def answer_with_context(self, query: str, context: str) -> str:
        """Generate answer using the retrieved context."""
        prompt = self.create_prompt(query, context)
        answer = self.generate(prompt)
        answer = self._clean_answer(answer)
        return answer
    
    def _clean_answer(self, answer: str) -> str:
        """Remove any formatting artifacts from the LLM response."""
        answer = re.sub(r"\[/?INST\]", "", answer)
        answer = re.sub(r"</?s>", "", answer)
        return answer.strip()
    
    def is_refusal(self, answer: str) -> bool:
        """Check if LLM refused to answer."""
        refusal_phrases = [
            "not specified in the document",
            "cannot be answered based on the provided documents",
            "not found in the context",
            "not mentioned in the document"
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in refusal_phrases)

    def healthcheck(self) -> Optional[str]:
        """Small sanity check for connectivity and auth.

        Returns an error string if something looks wrong, else None.
        """
        if not self.hf_token:
            return "HF_TOKEN is missing"
        return None
