from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

class SynthesizerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def combine(self, retrieved_docs):
        context = "\n".join([doc["content"] for doc in retrieved_docs])
        prompt = f"Summarize and answer based on these company documents:\n{context}"
        response = self.llm.invoke(prompt)
        return {"answer": response.content, "citations": [d["source"] for d in retrieved_docs]}
