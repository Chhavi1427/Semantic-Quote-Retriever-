from quote_embedder.embedder import build_faiss_index
from rag_pipeline import RAGQuoteRetriever
import os

if __name__ == "__main__":
    if not os.path.exists("faiss_index.idx") or not os.path.exists("quote_texts.pkl"):
        print("🔧 FAISS index not found. Building index...")
        build_faiss_index()
    else:
        print("✅ FAISS index already exists. Skipping rebuild.")

    rag = RAGQuoteRetriever()

    question = "What is a good quote about success?"
    print(f"\n🧠 Question: {question}")
    print("💬 Answer:", rag.generate_answer(question))
