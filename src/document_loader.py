"""Load and parse PDF documents."""
import fitz  # PyMuPDF
from typing import List, Dict
import re
import logging


logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads and parses a single PDF file."""
    
    def __init__(self, pdf_path: str, doc_name: str):
        """pdf_path: where the PDF is, doc_name: what to call it (e.g. 'Apple 10-K')"""
        self.pdf_path = pdf_path
        self.doc_name = doc_name
        
    def load(self) -> List[Dict[str, any]]:
        """Extract text from PDF. Returns list of pages with text and metadata."""
        doc = fitz.open(self.pdf_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Clean text
            text = self._clean_text(text)
            
            if text.strip():  # Only add non-empty pages
                pages_data.append({
                    'text': text,
                    'page': page_num + 1,  # 1-indexed
                    'doc_name': self.doc_name,
                    'section': self._extract_section(text)
                })
        
        doc.close()
        return pages_data
    
    def _clean_text(self, text: str) -> str:
        """Remove extra whitespace and weird characters."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = text.replace('\x00', '')
        return text.strip()
    
    def _extract_section(self, text: str) -> str:
        """Try to find section name like 'Item 8' in the text."""
        # Look for "Item X" patterns
        match = re.search(r'Item\s+\d+[A-Z]?\.?\s+[A-Za-z\s]+', text[:500])
        if match:
            return match.group(0).strip()
        return "Unknown Section"


def load_documents(apple_pdf: str, tesla_pdf: str) -> List[Dict[str, any]]:
    """Load both PDFs and return all pages combined."""
    logger.info("Loading Apple 10-K")
    apple_loader = DocumentLoader(apple_pdf, "Apple 10-K")
    apple_pages = apple_loader.load()
    logger.info("Loaded %s pages from Apple 10-K", len(apple_pages))
    
    logger.info("Loading Tesla 10-K")
    tesla_loader = DocumentLoader(tesla_pdf, "Tesla 10-K")
    tesla_pages = tesla_loader.load()
    logger.info("Loaded %s pages from Tesla 10-K", len(tesla_pages))
    
    return apple_pages + tesla_pages
