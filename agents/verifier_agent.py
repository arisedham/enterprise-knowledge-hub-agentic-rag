import difflib

class VerifierAgent:
    def check(self, synthesized, retrieved_docs):
        context = " ".join([doc["content"] for doc in retrieved_docs])
        match = difflib.SequenceMatcher(None, context.lower(), synthesized["answer"].lower())
        confidence = match.ratio() * 100
        synthesized["confidence"] = confidence
        return synthesized
