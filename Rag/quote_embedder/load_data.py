from datasets import load_dataset

def get_quotes(limit=None):
  
    dataset = load_dataset("Abirate/english_quotes", split="train")

    # Extract quotes and remove duplicates
    quotes = list(set(dataset["quote"]))

    # Optionally limit number
    if limit:
        quotes = quotes[:limit]
    
    return quotes
