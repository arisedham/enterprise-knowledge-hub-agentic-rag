# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import pytest
# from agents.retriever_agent import RetrieverAgent
# from langchain_openai import OpenAIEmbeddings  # ✅ updated import

# # Load .env file
# from dotenv import load_dotenv
# load_dotenv()


# @pytest.fixture
# def retriever():
#     """
#     Initialize the retriever agent using the existing vectorstore.
#     Make sure your vectorstore/chroma directory exists and has embeddings.
#     """
#     return RetrieverAgent(source="internal")


# def test_vectorstore_exists():
#     """Check if Chroma vectorstore directory exists and is not empty."""
#     vectorstore_path = "vectorstore/chroma"
#     assert os.path.exists(vectorstore_path), f"❌ Vectorstore not found at '{vectorstore_path}'"
#     assert len(os.listdir(vectorstore_path)) > 0, f"❌ '{vectorstore_path}' is empty — please ingest documents first."
#     print(f"✅ Found vectorstore with {len(os.listdir(vectorstore_path))} files.")


# def test_retriever_initialization(retriever):
#     """Ensure RetrieverAgent initializes successfully."""
#     assert retriever is not None, "❌ RetrieverAgent failed to initialize."
#     print("✅ RetrieverAgent initialized successfully.")


# def test_retrieve_results(retriever):
#     """Run a test query and verify retrieval returns relevant results."""
#     query = "What is the company leave policy?"
#     results = retriever.retrieve(query)

#     assert isinstance(results, list), "❌ Retriever did not return a list."
#     assert len(results) > 0, "❌ No results retrieved. Ensure the vectorstore is built with embeddings."
#     assert "content" in results[0], "❌ Retrieved result missing 'content' field."
#     print("\n✅ Top retrieved content preview:")
#     print(results[0]["content"][:300])


# if __name__ == "__main__":
#     retriever = RetrieverAgent(source="internal")
#     test_retrieve_results(retriever)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from utils.embedder import Embedder
from langchain.schema import Document


class TestEmbedder(unittest.TestCase):

    def setUp(self):
        # Use HuggingFace by default for offline testing
        self.embedder = Embedder(model_name="huggingface")
        self.sample_texts = [
            "AI is transforming the future of technology.",
            "LangChain provides tools for building LLM-based applications."
        ]

    def test_clean_text(self):
        dirty_text = "  This   is   a   test! 🚀   "
        cleaned = self.embedder.clean_text(dirty_text)
        self.assertTrue("  " not in cleaned)
        self.assertFalse("🚀" in cleaned)

    def test_embed_text(self):
        embeddings = self.embedder.embed_text(self.sample_texts)
        self.assertEqual(len(embeddings), len(self.sample_texts))
        self.assertIsInstance(embeddings[0], list)
        self.assertGreater(len(embeddings[0]), 0)

    def test_embed_query(self):
        query = "What is LangChain?"
        embedding = self.embedder.embed_query(query)
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)

    def test_chunk_documents(self):
        documents = [Document(page_content="AI research focuses on machine learning and deep learning.")]
        chunks = self.embedder.chunk_documents(documents)
        self.assertTrue(len(chunks) > 0)
        self.assertIn("AI research", chunks[0].page_content)


if __name__ == "__main__":
    unittest.main()
