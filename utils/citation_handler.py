"""
utils/citation_handler.py
-------------------------
Handles citation formatting and display for retrieved chunks.
Useful for showing source attribution in RAG outputs.
"""

from typing import List, Dict
from langchain.schema import Document


class CitationHandler:
    def __init__(self, citation_style: str = "APA"):
        """
        Initialize citation handler.
        Supported styles: 'APA', 'simple'
        """
        self.citation_style = citation_style.lower()

    def format_citation(self, doc: Document, index: int = None) -> str:
        """
        Create a readable citation for a single Document.
        """
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", None)

        if self.citation_style == "apa":
            citation = f"{source}"
            if page:
                citation += f", p.{page}"
        else:
            citation = f"[{source}]"
            if page:
                citation += f"(p.{page})"

        if index is not None:
            citation = f"[{index + 1}] {citation}"

        return citation

    def attach_citations(self, answer: str, retrieved_docs: List[Document]) -> str:
        """
        Append citations to the answer text.
        Example: "...according to company policy [1][2]"
        """
        if not retrieved_docs:
            return answer

        citations = []
        for i, doc in enumerate(retrieved_docs):
            citations.append(self.format_citation(doc, i))

        citation_text = "\n\n📚 **References:**\n" + "\n".join(citations)
        return answer.strip() + "\n" + citation_text

    def extract_sources(self, docs: List[Document]) -> List[Dict[str, str]]:
        """
        Extract structured list of sources for logging or UI display.
        """
        sources = []
        for i, doc in enumerate(docs):
            sources.append({
                "index": i + 1,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "-"),
                "preview": doc.page_content[:200] + "..."
            })
        return sources
