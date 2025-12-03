# app.py
import os
import lancedb
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

# Initialize models
print("Loading models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Free embedding model
chat_model = pipeline(
    "text-generation", 
    model="microsoft/DialoGPT-medium",
    tokenizer="microsoft/DialoGPT-medium",
    device=0 if torch.cuda.is_available() else -1
)

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


def generate_response_simple(query, contexts):
    """Generate a simple response based on context"""
    if not contexts:
        return "I couldn't find relevant information in the documents to answer your question."
    
    # Extract key information from contexts
    context_text = "\n\n".join([ctx["text"][:300] for ctx in contexts[:2]])
    
    # Create a simple template-based response
    response = f"Based on the documents, here's what I found:\n\n{context_text}"
    
    # Try to make it more conversational
    if "what" in query.lower() or "how" in query.lower():
        response = f"To answer your question: {response}"
    elif "is" in query.lower() or "are" in query.lower():
        response = f"According to the documents: {response}"
    
    return response[:800]  # Limit response length


class QueryIn(BaseModel):
    query: str
    n_results: int = 5


def build_prompt(query, contexts):
    system = "You are a helpful assistant. Use the context and cite sources as [0], [1], etc."

    ctx_texts = []
    for i, ctx in enumerate(contexts):
        source = ctx["metadata"].get("source", "unknown")
        snippet = ctx["text"][:1200]
        ctx_texts.append(f"[{i}] Source: {source}\n{snippet}")

    context_block = "\n\n---\n\n".join(ctx_texts)
    return f"{system}\n\nContext:\n{context_block}\n\nQuery: {query}\n\nAnswer:"


@app.post("/chat")
async def chat(q: QueryIn):
    # embed query using Hugging Face
    emb = embedding_model.encode(q.query).tolist()

    # vector search
    results = (
        table.search(emb)
        .limit(q.n_results)
        .select(["id", "text", "metadata"])
        .to_list()
    )

    contexts = results

    # Build context for the model
    prompt = build_prompt(q.query, contexts)
    
    # Generate response using Hugging Face
    try:
        # For DialoGPT, we need a simpler approach
        response_text = generate_response_simple(q.query, contexts)
        answer = response_text
    except Exception as e:
        answer = f"I found relevant information in the documents, but encountered an error generating a response: {str(e)}. Here are the key points from the context: {contexts[0]['text'][:200] if contexts else 'No relevant context found.'}..."
    
    sources = [c["metadata"].get("source", "unknown") for c in contexts]

    return {"answer": answer, "sources": sources}
