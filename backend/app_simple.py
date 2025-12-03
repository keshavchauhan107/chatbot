# app_simple.py - Offline version using TF-IDF
import os
import lancedb
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

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

# Load TF-IDF vectorizer
vectorizer = None
documents = []

# Check if we have pre-computed data
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('documents.pkl', 'rb') as f:
        documents = pickle.load(f)
    print("Loaded pre-computed TF-IDF data")
except:
    print("No pre-computed data found. Run ingest.py first.")

db = lancedb.connect(DB_DIR)
if COLLECTION_NAME not in db.table_names():
    raise RuntimeError("Run ingest.py first – no LanceDB collection found!")

table = db.open_table(COLLECTION_NAME)


class QueryIn(BaseModel):
    query: str
    n_results: int = 5


def search_documents(query, n_results=5):
    """Simple TF-IDF based search"""
    if not vectorizer or not documents:
        return []
    
    # Vectorize the query
    query_vec = vectorizer.transform([query])
    
    # Get all document vectors
    doc_vecs = vectorizer.transform(documents)
    
    # Calculate similarities
    similarities = cosine_similarity(query_vec, doc_vecs).flatten()
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:n_results]
    
    # Get results from LanceDB
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Minimum similarity threshold
            # Find corresponding document in LanceDB
            doc_results = table.search([0] * 300).limit(1000).to_list()
            if idx < len(doc_results):
                results.append(doc_results[idx])
    
    return results


@app.post("/chat")
async def chat(q: QueryIn):
    try:
        # Get all documents from LanceDB without vector search
        all_results = table.to_pandas().to_dict('records')
        
        # Simple keyword-based search
        query_lower = q.query.lower()
        
        # Find matching documents
        matching_results = []
        for result in all_results:
            text_lower = result['text'].lower()
            # Check for keyword matches
            if (any(word in text_lower for word in query_lower.split()) or
                query_lower in text_lower):
                matching_results.append(result)
        
        # If no matches, return all available documents for browsing
        if not matching_results:
            # For exploratory queries like "hi", "available docs", show what we have
            if len(query_lower) < 4 or query_lower in ['hi', 'hello', 'available', 'docs', 'help', 'what']:
                matching_results = all_results[:3]  # Show first few documents
            
        # Limit results
        results = matching_results[:q.n_results] if matching_results else all_results[:q.n_results]
        
        if not results:
            return {
                "answer": f"I couldn't find any documents. Please make sure documents are ingested properly.",
                "sources": []
            }
        
        # Generate response
        context_text = ""
        sources = []
        for i, result in enumerate(results[:3]):
            sources.append(result["metadata"].get("source", "unknown"))
        
        # Create a helpful response based on query type
        if query_lower in ['hi', 'hello', 'help', 'hey']:
            answer = f"Hello! 👋 I'm your RAG bot assistant. I can help you find information from your document collection.\n\n"
            answer += f"**Available Documents:** {', '.join(set(sources))}\n\n"
            answer += "**What I can do:**\n"
            answer += "• Answer questions about your documents\n"
            answer += "• Search for specific information\n" 
            answer += "• Provide summaries and insights\n\n"
            answer += "Try asking me something like:\n"
            answer += "• 'What can you do?'\n"
            answer += "• 'Tell me about the sample document'\n"
            answer += "• 'What information is available?'"
            
        elif 'what can you do' in query_lower or 'capabilities' in query_lower or 'features' in query_lower:
            # Extract key features from documents
            features_found = []
            for result in results:
                text = result['text'].lower()
                if 'feature' in text or 'capabilities' in text or 'can' in text:
                    # Extract relevant sentences
                    sentences = result['text'].split('.')
                    for sentence in sentences:
                        if any(word in sentence.lower() for word in ['feature', 'can', 'capability', 'provide', 'help']):
                            if len(sentence.strip()) > 10:
                                features_found.append(sentence.strip())
            
            answer = "Based on your documents, here are my capabilities:\n\n"
            if features_found:
                for i, feature in enumerate(features_found[:5], 1):
                    answer += f"{i}. {feature}\n"
            else:
                answer += "• **Document Search**: Find information across your document collection\n"
                answer += "• **Question Answering**: Get answers based on document content\n" 
                answer += "• **Content Retrieval**: Access specific parts of your documents\n"
                answer += "• **Knowledge Base**: Built from your uploaded documents\n"
            
        elif 'available' in query_lower or 'docs' in query_lower or 'documents' in query_lower:
            answer = f"📚 **Available Documents:** {', '.join(set(sources))}\n\n"
            answer += "**Document Preview:**\n"
            for i, result in enumerate(results[:2], 1):
                preview = result['text'][:200].strip()
                if not preview.endswith('.'):
                    preview = preview.rsplit('.', 1)[0] + '.' if '.' in preview else preview
                answer += f"{i}. {preview}...\n\n"
                
        else:
            # For specific queries, provide relevant content
            answer = f"Here's what I found about '{q.query}':\n\n"
            relevant_content = []
            for result in results[:3]:
                # Extract most relevant sentences
                sentences = result['text'].split('.')
                for sentence in sentences:
                    if any(word in sentence.lower() for word in query_lower.split() if len(word) > 2):
                        relevant_content.append(sentence.strip())
                        
            if relevant_content:
                for i, content in enumerate(relevant_content[:5], 1):
                    if len(content) > 10:
                        answer += f"• {content}\n"
                answer += "\n"
            else:
                # Fallback to document previews
                for i, result in enumerate(results[:2], 1):
                    preview = result['text'][:250].strip()
                    answer += f"**Excerpt {i}:** {preview}...\n\n"
        
        answer += f"\n📄 **Sources:** {', '.join(set(sources))}"
        
        return {"answer": answer, "sources": list(set(sources))}
        
    except Exception as e:
        return {
            "answer": f"Sorry, there was an error processing your request. Error: {str(e)}",
            "sources": []
        }