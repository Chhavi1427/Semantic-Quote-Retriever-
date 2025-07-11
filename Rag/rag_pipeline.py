from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

class RAGQuoteRetriever:
    def __init__(self,
                 embed_model="sentence-transformers/all-MiniLM-L6-v2",
                 gen_model="google/flan-t5-base"):
        self.embedder = SentenceTransformer(embed_model)

        try:
            self.index = faiss.read_index("faiss_index.idx")
            with open("quote_texts.pkl", "rb") as f:
                self.quotes = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Missing FAISS index or quote_texts.pkl. Please run build_faiss_index().")

        self.generator = pipeline("text2text-generation",
                                  model=gen_model,
                                  device=-1)  # CPU

    def retrieve(self, query, top_k=3):
        q_embed = self.embedder.encode([query])
        scores, indices = self.index.search(q_embed, top_k)
        return [self.quotes[i] for i in indices[0]]

    def generate_answer(self, query):
        context = " ".join(self.retrieve(query))
        prompt = f"Given these quotes: {context}\nAnswer the question: {query}"
        return self.generator(prompt, max_new_tokens=80)[0]['generated_text']
