# app.py
import os
import torch
import lancedb
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize free models
print("Loading embedding model...")
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Smaller model

DB_DIR = "lancedb_store"
COLLECTION_NAME = "docs"

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML file at root
@app.get("/")
async def serve_index():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

db = lancedb.connect(DB_DIR)
if COLLECTION_NAME not in db.table_names():
    raise RuntimeError("Run ingest.py first – no LanceDB collection found!")

table = db.open_table(COLLECTION_NAME)


class QueryIn(BaseModel):
    query: str
    n_results: int = 5


def build_prompt(query, contexts):
    """Build a simple response based on context"""
    if not contexts:
        return f"I couldn't find relevant information to answer: {query}"
    
    # Extract key information from contexts
    context_text = ""
    sources = []
    for i, ctx in enumerate(contexts[:3]):
        context_text += f"Context {i+1}: {ctx['text'][:400]}...\n\n"
        sources.append(ctx["metadata"].get("source", "unknown"))
    
    # Create a simple response
    response = f"Based on the available documents, here's what I found regarding '{query}':\n\n{context_text}"
    response += f"\nSources: {', '.join(set(sources))}"
    
    return response


@app.post("/chat")
async def chat(q: QueryIn):
    # embed query using sentence transformers
    emb = embedding_model.encode(q.query).tolist()

    # vector search
    results = (
        table.search(emb)
        .limit(q.n_results)
        .select(["id", "text", "metadata"])
        .to_list()
    )

    contexts = results

    # Generate simple response
    answer = build_prompt(q.query, contexts)
    sources = [c["metadata"].get("source", "unknown") for c in contexts]

    return {"answer": answer, "sources": list(set(sources))}