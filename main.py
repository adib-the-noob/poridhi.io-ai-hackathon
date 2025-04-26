import chromadb
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
app = FastAPI()

client = chromadb.HttpClient(host="54.169.92.49", port=8000)

# Define your Chroma collection
collection = client.get_or_create_collection("product_collection")

# Sample response model
class SearchResult(BaseModel):
    title: str
    description: str
    price: str

# Intent-based search function (you would implement your own search logic here)
def search_intent(query: str, intent: Optional[str] = None) -> List[SearchResult]:
    # For demonstration purposes, we're using static data
    # Replace this with your actual search logic
    results = collection.query(
        query_texts=[query], # Chroma will embed this for you
        n_results=6 # how many results to return
    )
    
    # Filter search results based on intent if provided
    if intent:
        results = [item for item in results if intent.lower() in item["title"].lower()]

    print("Results:", json.dumps(results, indent=2))
    # return [SearchResult(**item) for item in results if query.lower() in item["title"].lower()]
    return results

@app.get("/search")
async def search(query: str, intent: Optional[str] = Query(None, description="Intent for refined search")):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    results = search_intent(query, intent)
    if not results:
        raise HTTPException(status_code=404, detail="No results found")
    
    return results

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
