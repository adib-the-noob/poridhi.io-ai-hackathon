import requests
import numpy as np

def get_embedding(text, model="llama3.2"):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": text}
    )
    return response.json()["embedding"]

# List of texts to embed
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning models transform input data into numerical representations",
    "Semantic search uses vector embeddings to find relevant documents"
]

# Get embeddings for all texts
embeddings = [get_embedding(text) for text in texts]

# Convert to numpy array for easier manipulation
embeddings_array = np.array(embeddings)
print(embeddings_array)
print(f"Embeddings shape: {embeddings_array.shape}")