from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from quote_embedder.load_data import get_quotes

def build_faiss_index(model_name="all-MiniLM-L6-v2"):
    quotes = get_quotes()

    model = SentenceTransformer(model_name)
    embeddings = model.encode(quotes, convert_to_numpy=True)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index and quotes
    faiss.write_index(index, "faiss_index.idx")
    with open("quote_texts.pkl", "wb") as f:
        pickle.dump(quotes, f)

    print("FAISS index and quotes saved.")
