"""
utils/loader.py
---------------
Handles loading of PDF and text files for the RAG pipeline.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document


def load_single_file(file_path: str):
    """
    Load a single file (PDF or TXT).
    Returns a list of LangChain Document objects.
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return documents


def load_from_folder(folder_path: str):
    """
    Load all supported files (PDF, TXT, MD) from a folder.
    Returns a combined list of LangChain Document objects.
    """
    all_docs = []
    for file in os.listdir(folder_path):
        if file.lower().endswith((".pdf", ".txt", ".md")):
            file_path = os.path.join(folder_path, file)
            docs = load_single_file(file_path)
            all_docs.extend(docs)
    return all_docs


def prepare_metadata(docs, source_name: str = None):
    """
    Add metadata such as source or file name to documents.
    """
    for doc in docs:
        if source_name:
            doc.metadata["source"] = source_name
    return docs


def summarize_loaded_docs(docs):
    """
    Simple summary for debugging or logs.
    """
    total_chars = sum(len(d.page_content) for d in docs)
    return {
        "count": len(docs),
        "total_characters": total_chars,
        "avg_length": total_chars / len(docs) if docs else 0,
    }
