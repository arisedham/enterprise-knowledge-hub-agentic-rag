import streamlit as st
from agents.orchestrator_agent import OrchestratorAgent

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Enterprise Knowledge Hub", layout="wide")

st.title("🏢 Enterprise Knowledge Hub – Agentic RAG Demo")

query = st.text_input("Ask a question about company policy:")
if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        orchestrator = OrchestratorAgent()
        result = orchestrator.run(query)

        st.markdown("### 🧠 Answer")
        st.markdown(result["answer"])

        st.markdown("---")
        st.markdown("**Confidence:** {:.1f}%".format(result["confidence"]))
        st.markdown("**Sources:**")
        for c in result["citations"]:
            st.markdown(f"- {c}")
