from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

class RetrieverAgent:
    def __init__(self, source="internal"):
        self.source = source
        self.db = Chroma(persist_directory="vectorstore/chroma", embedding_function=OpenAIEmbeddings())

    def retrieve(self, query):
        docs = self.db.similarity_search_with_score(query, k=3)
        results = [{"content": d.page_content, "source": d.metadata.get("source")} for d, _ in docs]
        return results
