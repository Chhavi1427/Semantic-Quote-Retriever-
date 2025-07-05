import streamlit as st
from rag_pipeline import RAGQuoteRetriever

rag = RAGQuoteRetriever()

st.title("Semantic Quote Retriever 🧠💬")

question = st.text_input("Ask something motivational or philosophical:")

if question:
    st.subheader("Answer")
    st.write(rag.generate_answer(question))

    st.subheader("Top Quotes")
    for q in rag.retrieve(question):
        st.write(f"• {q}")
