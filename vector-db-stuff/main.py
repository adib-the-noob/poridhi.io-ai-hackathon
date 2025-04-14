import chromadb
from pprint import pprint

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="my_collection")

collection.add(
    documents=[
        "As a Programmer, You are a problem solver.",
        "Software development is a creative process.",
    ],
    ids=[
        "id1",
        "id2",
    ],
)


# Query 1
results = collection.query(
    query_texts=['software is eating the world'],
    n_results=2
)

pprint(results)
# {
#     'data': None,
#     'distances': [[0.12731888890266418 -> this is much closer!, 1.6445209980010986]],
#     'documents': [['Hello, world!', 'OpenAI makes powerful AI tools.']],
#     'embeddings': None,
#     'ids': [['id1', 'id2']],
#     'included': ['metadatas', 'documents', 'distances'],
#     'metadatas': [[None, None]],
#     'uris': None
# }

print('------')

# Query 2
result2 = collection.query(
    query_texts=["Hello, world"],
    n_results=2,
    where_document={
        '$contains': "World"
    }
)

pprint(result2)
print('------')