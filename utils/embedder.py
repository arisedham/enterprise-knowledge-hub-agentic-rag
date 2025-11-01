"""
utils/embedder.py
-----------------
Handles text chunking and embedding for RAG pipelines.
Supports OpenAI or HuggingFace embedding models.
"""

import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


class Embedder:
    def __init__(self, model_name="openai", chunk_size=800, chunk_overlap=100):
        """
        Initialize Embedder.
        model_name: 'openai' or 'huggingface'
        chunk_size: max tokens per chunk
        chunk_overlap: overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name.lower()

        if self.model_name == "openai":
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        elif self.model_name == "huggingface":
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError("Unsupported embedding model. Use 'openai' or 'huggingface'.")

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters, extra spaces, and artifacts.
        """
        text = re.sub(r'\s+', ' ', text)
        text = text.encode("ascii", "ignore").decode()
        return text.strip()

    def chunk_documents(self, documents):
        """
        Split LangChain Document objects into smaller chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunked_docs = text_splitter.split_documents(documents)
        return chunked_docs

    def embed_text(self, texts):
        """
        Generate embeddings for multiple text strings.
        """
        cleaned_texts = [self.clean_text(t) for t in texts]
        return self.embedding_model.embed_documents(cleaned_texts)

    def embed_query(self, query):
        """
        Generate embedding for a single query (for similarity search).
        """
        cleaned_query = self.clean_text(query)
        return self.embedding_model.embed_query(cleaned_query)
