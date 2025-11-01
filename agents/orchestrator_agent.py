from agents.router_agent import RouterAgent
from agents.synthesizer_agent import SynthesizerAgent
from agents.verifier_agent import VerifierAgent

class OrchestratorAgent:
    def __init__(self):
        self.router = RouterAgent()
        self.synthesizer = SynthesizerAgent()
        self.verifier = VerifierAgent()

    def run(self, query):
        retriever_results = self.router.route(query)
        synthesized = self.synthesizer.combine(retriever_results)
        verified = self.verifier.check(synthesized, retriever_results)
        return verified
