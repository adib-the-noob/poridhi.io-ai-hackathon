import openai

# Set your OpenAI API key
openai.api_key = "<api-key>"
def generate_embeddings(texts, model="text-embedding-ada-002"):

    try:
        response = openai.Embedding.create(
            input=texts,
            model=model
        )
        embeddings = [item['embedding'] for item in response['data']]
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    sample_texts = ["Hello, world!", "OpenAI makes powerful AI tools."]
    embeddings = generate_embeddings(sample_texts)
    if embeddings:
        for i, embedding in enumerate(embeddings):
            print(f"Embedding for text {i + 1}: {embedding[:5]}... (truncated)")