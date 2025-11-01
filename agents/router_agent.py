from agents.retriever_agent import RetrieverAgent

class RouterAgent:
    def __init__(self):
        self.internal_retriever = RetrieverAgent(source="internal")
        self.external_retriever = RetrieverAgent(source="external")

    def route(self, query):
        if "policy" in query.lower() or "leave" in query.lower():
            return self.internal_retriever.retrieve(query)
        else:
            return self.external_retriever.retrieve(query)
