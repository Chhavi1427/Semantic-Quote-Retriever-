import streamlit as st
import os
from quote_embedder.embedder import build_faiss_index
from rag_pipeline import RAGQuoteRetriever

if not os.path.exists("faiss_index.idx") or not os.path.exists("quote_texts.pkl"):
    st.warning("Building FAISS index... please wait ⏳")
    build_faiss_index()
    st.success("Index built! Ready to go 🚀")

rag = RAGQuoteRetriever()

st.title("Semantic Quote Retriever 🧠💬")

question = st.text_input("Ask something motivational or philosophical:")

if question:
    st.subheader("Answer")
    st.write(rag.generate_answer(question))

    st.subheader("Top Quotes")
    for q in rag.retrieve(question):
        st.write(f"• {q}")

