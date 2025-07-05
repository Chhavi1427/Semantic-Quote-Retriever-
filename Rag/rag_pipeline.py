from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch
import os

class RAGQuoteRetriever:
    def __init__(self,
                 embed_model="sentence-transformers/all-MiniLM-L6-v2",
                 gen_model="google/flan-t5-base"):
        # ✅ Avoid meta tensor error (set default device + disable sharing)
        os.environ["TORCH_LOAD_DISABLE_SHARING"] = "1"
        torch.set_default_device("cpu")

        # ✅ Load SentenceTransformer safely on CPU
        self.embedder = SentenceTransformer(embed_model)

        # ✅ Load FAISS index
        self.index = faiss.read_index("faiss_index.idx")

        # ✅ Load original quote texts
        with open("quote_texts.pkl", "rb") as f:
            self.quotes = pickle.load(f)

        # ✅ Load generator pipeline (on CPU)
        self.generator = pipeline("text2text-generation",
                                  model=gen_model,
                                  device=-1)  # -1 = CPU

    def retrieve(self, query, top_k=3):
        # Embed the query and retrieve top matching quotes
        q_embed = self.embedder.encode([query])
        scores, indices = self.index.search(q_embed, top_k)
        return [self.quotes[i] for i in indices[0]]

    def generate_answer(self, query):
        # Generate an answer using the retrieved quotes as context
        context = " ".join(self.retrieve(query))
        prompt = f"Given these quotes: {context}\nAnswer the question: {query}"
        return self.generator(prompt, max_new_tokens=80)[0]['generated_text']
